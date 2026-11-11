# SPDX-License-Identifier: Apache-2.0
"""Fixed-batch evaluation for ksdraft checkpoints (train/val CE, block CE).

The trainer's logged loss is a single random batch, so checkpoint comparison
needs a fixed validation set. This script samples deterministic validation
windows from a token stream and reports next-token CE and the parallel block
head's CE at each offset.

  python scripts/eval_draft.py --config configs/flint.yaml \
      --checkpoint runs/warmup_flint5/ckpt_002000.pt \
      --corpus /home/curved/SDGraft/data/flint_longctx/val_tokens.pt \
      --block-n 4 --batches 40
"""

from __future__ import annotations

import argparse

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flint.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ksdraft.cli import _load_model
    from ksdraft.train import WindowSampler

    model = _load_model(argparse.Namespace(
        config=args.config, checkpoint=args.checkpoint)).to(args.device).eval()
    ids = torch.load(args.corpus, map_location="cpu", weights_only=True)
    sampler = WindowSampler(ids, seq_len=args.seq_len + args.block_n, seed=1234)
    ce_tot = block_tot = 0.0
    per_offset = [0.0] * args.block_n
    with torch.no_grad():
        for _ in range(args.batches):
            batch = sampler.batch(args.batch).to(args.device)
            ctx = batch[:, :args.seq_len]
            main = batch[:, :args.seq_len + 1]
            logits, _, blog = model.forward_block(ctx, args.block_n)
            ce = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), main[:, 1:].flatten())
            tgt = batch[:, args.seq_len:args.seq_len + args.block_n]
            per = [torch.nn.functional.cross_entropy(
                       blog[:, k].float(), tgt[:, k]) for k in range(args.block_n)]
            bce = torch.stack(per).mean()
            ce_tot += float(ce)
            block_tot += float(bce)
            for k, v in enumerate(per):
                per_offset[k] += float(v)
    n = args.batches
    print(f"ckpt {args.checkpoint}")
    print(f"  next-token CE     {ce_tot / n:.4f}")
    print(f"  block CE (mean)   {block_tot / n:.4f}")
    print("  block CE per offset " +
          " ".join(f"{k}:{v / n:.3f}" for k, v in enumerate(per_offset)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
