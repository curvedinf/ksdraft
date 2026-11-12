# SPDX-License-Identifier: Apache-2.0
"""Prepare the on-policy shards for training: validate, split, save.

The builder writes rolling full snapshots (each shard contains every sample
so far), so the newest shard is the corpus. This step verifies the document
boundaries, splits by document (never mid-turn), and writes the dict form
WindowSampler expects:

    {"tokens": int64[N], "boundaries": int64[D]}

  python scripts/prepare_corpus.py --shards runs/onpolicy \
      --out runs/onpolicy_ready
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def latest_shard(root: Path) -> Path:
    shards = sorted(root.glob("shard_*.pt"))
    if not shards:
        raise SystemExit(f"no shard_*.pt under {root}")
    return shards[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, action="append",
                    help="shard directory (newest shard) or .pt file; repeat "
                         "to merge tiers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--tokenizer", default=None,
                    help="optional: verify every boundary is a chat turn")
    args = ap.parse_args()

    parts = []
    for raw in args.shards:
        p = Path(raw)
        shard = latest_shard(p) if p.is_dir() else p
        data = torch.load(shard, map_location="cpu", weights_only=True)
        parts.append((shard, data))
    token_parts, bound_parts, meta = [], [], []
    offset = 0
    for shard, data in parts:
        t = data["tokens"].long()
        b = data["boundaries"].long()
        print(f"{shard.name}: {t.numel()} tokens, {b.numel()} docs",
              flush=True)
        token_parts.append(t)
        bound_parts.append(b + offset)
        meta.extend(data.get("meta") or [])
        offset += t.numel()
    tokens = torch.cat(token_parts)
    bounds = torch.cat(bound_parts)
    print(f"merged: {tokens.numel()} tokens, {bounds.numel()} docs",
          flush=True)

    if bounds.numel() and int(bounds[0]) != 0:
        raise SystemExit("first boundary is not 0")
    if bounds.numel() > 1 and not bool((bounds[1:] > bounds[:-1]).all()):
        raise SystemExit("boundaries are not strictly increasing")
    if bounds.numel() and int(bounds[-1]) >= tokens.numel():
        raise SystemExit("boundary outside the token stream")

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            trust_remote_code=False)
        bad = 0
        for b in bounds.tolist()[:2000]:
            head = tok.decode(tokens[b:b + 2].tolist())
            if not (head.startswith("<|user|>") or
                    head.startswith("<|system|>")):
                bad += 1
        print(f"boundary check: {bad}/2000 not a chat turn", flush=True)
        if bad:
            raise SystemExit("boundaries do not align with chat turns")

    n_docs = bounds.numel()
    n_val = max(1, int(n_docs * args.val_frac))
    cut_doc = n_docs - n_val
    cut = int(bounds[cut_doc])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = {"tokens": tokens[:cut].clone(),
             "boundaries": bounds[:cut_doc].clone()}
    val = {"tokens": tokens[cut:].clone(),
           "boundaries": bounds[cut_doc:].clone() - cut}
    torch.save(train, out / "train_tokens.pt")
    torch.save(val, out / "val_tokens.pt")
    summary = {"sources": [str(p[0]) for p in parts],
               "tokens": int(tokens.numel()),
               "docs": n_docs, "train_tokens": int(train["tokens"].numel()),
               "val_tokens": int(val["tokens"].numel()),
               "train_docs": int(train["boundaries"].numel()),
               "val_docs": int(val["boundaries"].numel()),
               "kinds": data.get("meta") and
                        {k: sum(1 for m in meta if m["kind"] == k)
                         for k in {m["kind"] for m in meta}}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
