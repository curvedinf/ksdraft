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
    ap.add_argument("--train-dtype", default="fp16",
                    choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--embedding-init", default=None)
    ap.add_argument("--transfer-space", default=None)
    ap.add_argument("--transfer-init", action="store_true")
    ap.add_argument("--freeze-embeddings", action="store_true")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--block-n", type=int, default=4)
    ap.add_argument("--block-kl", type=float, default=1.0,
                    help="forward-KL weight on the block offsets (DFlash)")
    ap.add_argument("--block-decay", type=float, default=0.8,
                    help="DFlash fixed exponential position weights")
    ap.add_argument("--block-ce", type=float, default=0.1,
                    help="hard-CE auxiliary on the block offsets (standard: "
                         "small, the forward KL is primary)")
    ap.add_argument("--selector-weight", type=float, default=1.0,
                    help="DFlash2 pairwise path-selector CE weight")
    ap.add_argument("--selector-topk", type=int, default=16)
    ap.add_argument("--transfer-gamma", type=float, default=2.0)
    ap.add_argument("--spine-sub-prob", type=float, default=0.1)
    args = ap.parse_args()

    from ksdraft.cli import _load_target_config
    from ksdraft.model import KSDraftModel
    from ksdraft.train import transfer_alignment

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    windows = cache["windows"].long()
    topk_ids = cache["topk_ids"].long()
    topk_probs = cache["topk_probs"].float()
    # partial mining checkpoints zero-pad beyond the finished prefix; older
    # checkpoints predate the "filled" key, so infer the prefix from the data
    if "filled" in cache:
        filled = int(cache["filled"])
    else:
        filled = int((topk_ids.abs().sum(-1) > 0).any(-1).sum())
    windows = windows[:filled]
    topk_ids = topk_ids[:filled]
    topk_probs = topk_probs[:filled]
    n = windows.shape[0]
    print(f"[distill] {n} windows, seq_len={windows.shape[1] - 1}")

    cfg, _tspec, trspec = _load_target_config(args.config)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    emb = (torch.load(args.embedding_init, map_location="cpu",
                      weights_only=True) if args.embedding_init else None)
    ts_space = None
    if args.transfer_space:
        ts_space = torch.load(args.transfer_space, map_location="cpu",
                              weights_only=True)
    model = KSDraftModel(cfg, teacher_dim=0, embedding_init=emb,
                         transfer_space=ts_space).to(dev).train()
    if args.freeze_embeddings:
        model.embedding.weight.requires_grad_(False)
        print("[distill] embedding/head frozen", flush=True)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, _ = model.load_state_dict(ck["model"], strict=False)
        print(f"[distill] init from {args.init} (step {ck.get('step')}"
              f"{f', {len(missing)} new surfaces' if missing else ''})")
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        opt = torch.optim.AdamW(params, lr=args.lr,
                                weight_decay=0.01, fused=True)
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(params, lr=args.lr,
                                weight_decay=0.01)
    self_seq_len = min(16, windows.shape[1] - 1)
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
        from ksdraft.kernels.fused_train import (block_position_weights,
                                                 linear_ce, linear_soft_ce,
                                                 wtv_from_logp)
        amp = (torch.autocast("cuda", dtype=torch.float16)
               if args.device != "cpu" and args.train_dtype == "fp16"
               else torch.autocast("cuda", enabled=False))
        with amp:
            sub_p = args.spine_sub_prob * max(
                0.0, 1.0 - step / max(1.0, 0.5 * args.steps))
            h, feats = model.trunk(w[:, :-1], sub_prob=sub_p)
        d = h.shape[-1]
        weight = model.head.weight
        k = ids.shape[-1]
        ce, _ = linear_ce(h.reshape(-1, d), weight, w[:, 1:].reshape(-1),
                          None, args.chunk)
        kl, logp = linear_soft_ce(h.reshape(-1, d), weight,
                                  ids.reshape(-1, k), probs.reshape(-1, k),
                                  None, args.chunk)
        wtv = wtv_from_logp(logp, probs.reshape(-1, k))
        ts = transfer_alignment(feats, model.transfer_C, w[:, 1:],
                                args.transfer_gamma)
        loss = ce + args.transfer * ts + args.kl * kl + args.wtv * wtv
        parts = {"ce": float(ce.detach()), "transfer": float(ts.detach()),
                 "kl": float(kl.detach()), "wtv": float(wtv.detach())}
        # DFlash-standard block objective: the served proposal surface is
        # anchor + block_delta[k]; supervise offset k against the teacher
        # distribution at its true position, with exponential decay.
        T = w.shape[1] - 1
        ctx_len = min(self_seq_len, T)
        anchor = ctx_len - 1
        bn = min(args.block_n, T - anchor)
        if bn > 0 and (args.block_kl > 0 or args.block_ce > 0
                       or args.selector_weight > 0):
            hs = model.block_conv(
                h[:, anchor].unsqueeze(1)
                + model.block_delta[:bn].unsqueeze(0))
            ids_off = torch.stack(
                [ids[:, anchor + j] for j in range(bn)], 1).reshape(-1, k)
            p_off = torch.stack(
                [probs[:, anchor + j] for j in range(bn)], 1).reshape(-1, k)
            bw = block_position_weights(bn, args.block_decay, h.device)
            w_off = bw.repeat(h.shape[0])
            if args.block_kl > 0:
                bkl, _ = linear_soft_ce(hs.reshape(-1, d), weight, ids_off,
                                        p_off, w_off, args.chunk)
                loss = loss + args.block_kl * bkl
                parts["block_kl"] = float(bkl.detach())
            if args.block_ce > 0:
                tgt_off = torch.stack(
                    [w[:, anchor + 1 + j] for j in range(bn)], 1).reshape(-1)
                bce, _ = linear_ce(hs.reshape(-1, d), weight, tgt_off, w_off,
                                   args.chunk)
                loss = loss + args.block_ce * bce
                parts["block_ce"] = float(bce.detach())
            if args.selector_weight > 0:
                sel = model.selector_loss(hs, w, anchor, bn,
                                          topk=args.selector_topk)
                loss = loss + args.selector_weight * sel
                parts["selector"] = float(sel.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"[distill {step:4d}] loss {loss.item():.4f} ce {parts['ce']:.4f} "
                  f"kl {parts.get('kl', 0):.4f} bkl {parts.get('block_kl', 0):.4f} "
                  f"sel {parts.get('selector', 0):.4f} "
                  f"wtv {parts.get('wtv', 0):.4f} "
                  f"({args.batch * (step + 1) / max(1e-9, el):.0f} win/s)",
                  flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save({"step": step, "model": model.state_dict(),
                        "train_cfg": vars(args), "config": cfg.summary()},
                       out / f"ckpt_{step:06d}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
