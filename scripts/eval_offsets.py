# SPDX-License-Identifier: Apache-2.0
"""Held-out next-token and per-draft-offset metrics for a checkpoint.

  python scripts/eval_offsets.py --checkpoint runs/phaseA_20m/ckpt_latest.pt \
      --cache runs/onpolicy_eval_cache20.pt --block-n 4 --seq-len 16
"""

from __future__ import annotations

import argparse
import json

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flint.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--block-n", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ksdraft.cli import _load_model
    from ksdraft.train import held_out_metrics

    model = _load_model(argparse.Namespace(
        config=args.config, checkpoint=args.checkpoint)).to(args.device).eval()
    c = torch.load(args.cache, map_location="cpu", weights_only=True)
    n = int(c.get("filled", c["windows"].shape[0]))
    tensors = (c["windows"][:n].long(), c["topk_ids"][:n].long(),
               c["topk_probs"][:n].float())
    amp = (torch.autocast("cuda", dtype=torch.float16)
           if args.device != "cpu" else torch.autocast("cuda", enabled=False))
    with torch.no_grad():
        out = held_out_metrics(model, tensors, block_n=args.block_n,
                               seq_len=args.seq_len, batch=args.batch,
                               device=args.device, amp=amp, limit=args.limit)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
