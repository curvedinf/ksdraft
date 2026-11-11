# SPDX-License-Identifier: Apache-2.0
"""Distill the draft from a mined teacher cache (CE + teacher-support CE + WTV).

The teacher cache holds window-aligned (tokens, top-k, probs) from
`scripts/mine_teacher.py`; the objective is the temp-1.0 distribution match
that governs serving acceptance.

  python scripts/distill_from_cache.py --config configs/flint.yaml \
      --cache runs/teacher_cache.pt --init runs/warmup_flint/ckpt_002000.pt \
      --out runs/distill --steps 4000 --batch 32
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flint.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--init", default=None, help="warmup checkpoint to start from")
    ap.add_argument("--out", default="runs/distill")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--transfer", type=float, default=0.5)
    ap.add_argument("--kl", type=float, default=0.2)
    ap.add_argument("--wtv", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ksdraft.cli import _load_target_config
    from ksdraft.model import KSDraftModel
    from ksdraft.train import distillation_losses

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    windows = cache["windows"].long()
    topk_ids = cache["topk_ids"].long()
    topk_probs = cache["topk_probs"].float()
    n = windows.shape[0]
    print(f"[distill] {n} windows, seq_len={windows.shape[1] - 1}")

    cfg, _tspec, trspec = _load_target_config(args.config)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    model = KSDraftModel(cfg, teacher_dim=0).to(dev).train()
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[distill] init from {args.init} (step {ck.get('step')})")
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01, fused=True)
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01)
    g = torch.Generator().manual_seed(11)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        if step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        else:
            p = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
        for grp in opt.param_groups:
            grp["lr"] = lr
        sel = torch.randint(0, n, (args.batch,), generator=g)
        w = windows[sel].to(dev)
        ids = topk_ids[sel].to(dev)
        probs = topk_probs[sel].to(dev)
        logits, feats = model(w[:, :-1])
        loss, parts = distillation_losses(
            logits, w, feats, model.embedding.weight, probs, ids,
            args.transfer, args.kl, args.wtv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"[distill {step:4d}] loss {loss.item():.4f} ce {parts['ce']:.4f} "
                  f"kl {parts.get('kl', 0):.4f} wtv {parts.get('wtv', 0):.4f} "
                  f"({args.batch * (step + 1) / max(1e-9, el):.0f} win/s)",
                  flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save({"step": step, "model": model.state_dict(),
                        "train_cfg": vars(args), "config": cfg.summary()},
                       out / f"ckpt_{step:06d}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
