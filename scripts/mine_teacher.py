# SPDX-License-Identifier: Apache-2.0
"""Mine teacher top-k signals for window-aligned corpus chunks.

Runs an in-process vLLM teacher (`OfflineLogitTeacher`) over consecutive
T+1-token windows of a token stream and writes one cache shard with the
windows plus their per-position top-k distributions (T=1, serving truncation).
The distill pass consumes this cache without needing the teacher resident.

  python scripts/mine_teacher.py --model <flint dir> --corpus <tokens.pt> \
      --out runs/teacher_cache.pt --windows 20000 --tp 4
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from ksdraft.teacher.offline import OfflineLogitTeacher

    tokens = torch.load(args.corpus, map_location="cpu", weights_only=True)
    if isinstance(tokens, dict):
        tokens = tokens.get("tokens", tokens.get("ids"))
    T = args.seq_len
    n_max = max(0, len(tokens) // T - 1)
    n = min(args.windows, n_max)
    if n <= 0:
        raise SystemExit("corpus too short for the requested window count")
    idx = (torch.arange(n)[:, None] * T + torch.arange(T + 1)[None, :])
    windows = tokens[idx].long()

    teacher = OfflineLogitTeacher(
        args.model, tensor_parallel_size=args.tp, topk=args.topk,
        serve_top_p=args.top_p, serve_top_k=args.top_k,
        max_model_len=max(4096, T + 2), gpu_memory_utilization=args.gpu_mem,
    )
    win_chunks, id_chunks, p_chunks = [], [], []
    t0 = time.time()
    for i in range(0, n, args.batch):
        w = windows[i:i + args.batch]
        sig = teacher.signals(w[:, :-1])
        win_chunks.append(w)
        id_chunks.append(sig.topk_ids)
        p_chunks.append(sig.topk_probs)
        done = min(i + args.batch, n)
        if done % (args.batch * 10) == 0 or done == n:
            rate = done / max(1e-9, time.time() - t0)
            print(f"[mine] {done}/{n} windows ({rate:.0f} win/s)", flush=True)
    torch.save({
        "windows": torch.cat(win_chunks),
        "topk_ids": torch.cat(id_chunks),
        "topk_probs": torch.cat(p_chunks),
        "seq_len": T,
        "topk": args.topk,
        "serve": {"top_p": args.top_p, "top_k": args.top_k},
    }, args.out)
    print(f"wrote {args.out} ({n} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
