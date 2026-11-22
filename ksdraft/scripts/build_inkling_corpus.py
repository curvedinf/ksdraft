# SPDX-License-Identifier: Apache-2.0
"""Tokenize a text corpus with the Inkling tokenizer (run under quant-venv
with the transformers fork on PYTHONPATH — see mine_inkling_signals.py).

Sources: local HF-cached datasets (evol-codealpaca-v1 = code instructions,
fineweb sample-10BT = web text) and/or plain text files (one doc per line
or paragraph blocks). Output: tokens.pt (int32), boundaries.pt (int64
document starts), meta.json.

  PYTHONPATH=/home/curved/models/inkling-transformers-src/src \
  ~/quant-venv/bin/python scripts/build_inkling_corpus.py \
      --dataset theblackcat102/evol-codealpaca-v1 --max-tokens 6000000 \
      --pack 12288 --out runs/inkling_corpus/long
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _doc_text(row):
    for key in ("text", "output", "instruction", "content", "code"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def iter_dataset(name, cache_dir):
    from datasets import load_dataset

    parts = name.split("::", 1)
    config = parts[1] if len(parts) > 1 else None
    name = parts[0]
    # non-streaming: local caches (evol-codealpaca) load whole; hub datasets
    # with "::" fall back to streaming to avoid full downloads
    streaming = config is not None
    ds = load_dataset(name, config, split="train", streaming=streaming)
    for row in ds:
        text = _doc_text(row)
        if text:
            yield text


def iter_texts(paths):
    for p in paths:
        for line in Path(p).read_text(errors="ignore").splitlines():
            if line.strip():
                yield line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/home/curved/models/Inkling-2-Mini-PTQR-W4A8-G128")
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--max-tokens", type=int, default=8_000_000)
    ap.add_argument("--pack", type=int, default=0,
                    help=">0: pack documents with EOS separators into "
                         "streams of at least this many tokens (long-context)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    rng = random.Random(args.seed)

    sources = []
    for name in args.dataset:
        sources.append(iter_dataset(name, None))
    for paths in ([args.text] if args.text else []):
        sources.append(iter_texts(paths))
    if not sources:
        raise SystemExit("no sources: pass --dataset and/or --text")
    # round-robin the sources so mixes stay balanced
    active = list(sources)
    docs = []
    while active:
        idx = rng.randrange(len(active))
        try:
            docs.append(next(active[idx]))
        except StopIteration:
            active.pop(idx)

    tokens, bounds = [], [0]
    packed, packed_len = [], 0

    def flush_doc(ids):
        nonlocal packed, packed_len
        if args.pack:
            packed.extend(ids)
            packed.append(eos)
            packed_len = len(packed)
            if packed_len >= args.pack:
                _emit(packed)
                packed, packed_len = [], 0
        else:
            _emit(ids + [eos])

    def _emit(ids):
        nonlocal tokens
        tokens.extend(ids)
        bounds.append(len(tokens))

    total = 0
    batch, batch_chars = [], 0
    for doc in docs:
        batch.append(doc)
        batch_chars += len(doc)
        if batch_chars >= 200_000:
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                if not ids:
                    bounds[-1] = len(tokens)
                    continue
                flush_doc(ids)
            total = len(tokens)
            batch, batch_chars = [], 0
            print(f"  {total:,} tokens, {len(bounds):,} docs", flush=True)
            if total >= args.max_tokens:
                break
    if batch:
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        for ids in enc:
            if ids:
                flush_doc(ids)
    if args.pack and packed:
        _emit(packed)

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = torch.tensor(tokens[:args.max_tokens + args.pack], dtype=torch.int32)
    b = torch.tensor([x for x in bounds if x <= t.numel()], dtype=torch.int64)
    torch.save(t, out / "tokens.pt")
    torch.save(b, out / "boundaries.pt")
    (out / "meta.json").write_text(json.dumps({
        "tokenizer": args.tokenizer, "num_tokens": int(t.numel()),
        "num_docs": int(b.numel() - 1), "pack": args.pack,
        "datasets": args.dataset, "texts": args.text,
        "eos": eos, "seed": args.seed}, indent=2))
    print(f"saved {t.numel():,} tokens / {b.numel() - 1:,} docs -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
