# SPDX-License-Identifier: Apache-2.0
"""Mine teacher top-k over HTTP from the served Flint recipe.

The in-process ``LLM()`` path cannot load the ``flint`` config type, so the
teacher cache is mined through the same OpenAI-compatible endpoint the draft
corpus builder uses. Each request sends one token window as the prompt with
``prompt_logprobs=K``; position i gives the distribution that predicts token
i+1, which is what the distill pass consumes. The serving truncation
(top-k then top-p, renormalized) is applied exactly as at sampling time.

Output format matches ``scripts/mine_teacher.py``:
    {"windows": [N, T+1], "topk_ids": [N, T, K], "topk_probs": [N, T, K],
     "seq_len": T, "topk": K, "serve": {...}}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


def parse_entry(entry, topk, serve_top_k, serve_top_p):
    if not entry:
        return None
    cand = sorted(entry.items(), key=lambda kv: kv[1]["logprob"],
                  reverse=True)[:topk]
    p = torch.tensor([math.exp(kv[1]["logprob"]) for kv in cand],
                     dtype=torch.float32)
    idx = torch.tensor([int(kv[0]) for kv in cand], dtype=torch.int64)
    if serve_top_k and len(p) > serve_top_k:
        p, idx = p[:serve_top_k], idx[:serve_top_k]
    if serve_top_p < 1.0 and len(p) > 1:
        c = torch.cumsum(p, 0)
        keep = int((c < serve_top_p).sum().item()) + 1
        keep = max(1, min(keep, len(p)))
        p, idx = p[:keep], idx[:keep]
    p = p / p.sum().clamp_min(1e-9)
    return idx, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8030/v1")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--model", default="flint")
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    data = torch.load(args.corpus, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        tokens = data.get("tokens", data.get("ids"))
        bounds = data.get("boundaries")
    else:
        tokens, bounds = data, None
    tokens = tokens.long()
    T = args.seq_len
    span = T + 1

    # Document-aligned windows: sample a start inside a document.
    g = torch.Generator().manual_seed(args.seed)
    if bounds is not None and len(bounds):
        starts = torch.unique(torch.cat([torch.zeros(1, dtype=torch.long),
                                         bounds.long()]))
        ends = torch.cat([starts[1:], torch.tensor([len(tokens)])])
        lens = ends - starts
        ok = lens >= span
        starts, lens = starts[ok], lens[ok]
        if starts.numel() == 0:
            raise SystemExit("no document long enough for the window")
        pick = torch.randint(0, starts.numel(), (args.windows,), generator=g)
        off = (torch.rand(args.windows, generator=g) *
               (lens[pick] - span + 1)).long()
        idx = starts[pick] + off
    else:
        idx = torch.randint(0, len(tokens) - span, (args.windows,),
                            generator=g)
    windows = tokens[idx[:, None] + torch.arange(span)[None, :]]

    # The served engine caps prompt_logprobs (20 on this build); the cache
    # keeps the configured width with zero padding beyond the returned set.
    req_k = min(args.topk, 20)

    def one(w):
        body = {"model": args.model, "prompt": w.tolist(),
                "max_tokens": 1, "temperature": 0.0,
                "prompt_logprobs": req_k}
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {args.api_key}"}
                        if args.api_key else {})})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        return out["choices"][0].get("prompt_logprobs") or []

    ids = torch.zeros(len(windows), T, args.topk, dtype=torch.int64)
    probs = torch.zeros(len(windows), T, args.topk, dtype=torch.float16)
    done = 0
    filled = 0
    t0 = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(one, windows[i]): i
                for i in range(len(windows))}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                plp = fut.result()
            except Exception as exc:
                print(f"[fail] {i}: {type(exc).__name__}: {exc}", flush=True)
                continue
            for t in range(T):
                parsed = parse_entry(plp[t + 1] if t + 1 < len(plp) else None,
                                     args.topk, args.top_k, args.top_p)
                if parsed is None:
                    continue
                ci, cp = parsed
                ids[i, t, :len(ci)] = ci
                probs[i, t, :len(cp)] = cp
            done += 1
            if any(ids[i, t, 0] != 0 for t in range(T)):
                filled += 1
            if done % (args.save_every // 4) == 0:
                rate = done / max(1e-9, time.time() - t0)
                print(f"[mine] {done}/{len(windows)} ({rate:.1f}/s)",
                      flush=True)
            if done % args.save_every == 0:
                torch.save({"windows": windows, "topk_ids": ids,
                            "topk_probs": probs, "filled": filled,
                            "seq_len": T,
                            "topk": args.topk,
                            "serve": {"top_p": args.top_p,
                                      "top_k": args.top_k}}, out_path)
    torch.save({"windows": windows, "topk_ids": ids, "topk_probs": probs,
                "filled": filled, "seq_len": T, "topk": args.topk,
                "serve": {"top_p": args.top_p, "top_k": args.top_k}},
               out_path)
    print(f"wrote {out_path} ({len(windows)} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
