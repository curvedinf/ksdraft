"""ksdraft command-line interface.

Commands (RESEARCH.md Section 12A):
  size    draft-shape budget from config (implemented)
  mine    corpus tiers + teacher mining            (staged in later milestones)
  train   staged training regime, Sections 11      (staged in later milestones)
  gate    PTQR battery + acceptance gates          (staged in later milestones)
  export  serving formats                          (staged in later milestones)
"""

from __future__ import annotations

import argparse
import sys

from .config import KSDraftConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ksdraft",
        description="Parallel block-drafting draft models (RWKV-8 Rosa + KS3).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_size = sub.add_parser("size", help="print the draft-shape parameter budget")
    p_size.add_argument("--layers", type=int, default=KSDraftConfig.layers)
    p_size.add_argument("--d-model", type=int, default=KSDraftConfig.d_model)
    p_size.add_argument("--head-size", type=int, default=KSDraftConfig.head_size)
    p_size.add_argument("--vocab", type=int, default=KSDraftConfig.vocab_size)
    p_size.add_argument("--rosa-streams", type=int, default=KSDraftConfig.rosa_streams)
    p_size.add_argument("--rosa-bits", type=int, default=KSDraftConfig.rosa_bits)
    p_size.add_argument("--ks3-charts", type=int, default=0, help="0 = density default (3.83 per width unit)")
    p_size.add_argument("--dtype", default="int8", choices=["int8", "bf16", "fp32"])

    for name, help_text in (
        ("mine", "build corpus tiers and mine teacher signals"),
        ("train", "run the staged training regime"),
        ("gate", "run PTQR fidelity + acceptance gates"),
        ("export", "export serving formats"),
    ):
        sub.add_parser(name, help=help_text)

    return parser


def _cmd_size(args: argparse.Namespace) -> int:
    cfg = KSDraftConfig(
        vocab_size=args.vocab,
        layers=args.layers,
        d_model=args.d_model,
        head_size=args.head_size,
        rosa_streams=args.rosa_streams,
        rosa_bits=args.rosa_bits,
        ks3_charts=args.ks3_charts,
        dtype=args.dtype,
    )
    s = cfg.summary()
    print(f"ksdraft shape budget (dtype={s['dtype']})")
    print(f"  layers            {s['layers']}")
    print(f"  d_model           {s['d_model']}  (head granularity: {s['n_heads']} x {s['head_size']})")
    print(f"  rosa              {s['rosa_streams']} streams x {s['rosa_bits']} bit(s)")
    print(f"  ks3 charts        {s['ks3_charts']}")
    print(f"  rosa/layer        {s['rosa_params_per_layer']:>15,}")
    print(f"  ks3/layer         {s['ks3_params_per_layer']:>15,}")
    print(f"  total/layer       {s['layer_params']:>15,}")
    print(f"  embedding (tied)  {s['embedding_params']:>15,}")
    print(f"  TOTAL             {s['total_params']:>15,}  (~{s['total_params'] / 1e9:.2f}B)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "size":
        return _cmd_size(args)
    print(
        f"'{args.command}' is staged for a later milestone (see RESEARCH.md); "
        "not implemented yet.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
