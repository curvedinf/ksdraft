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

    p_mine = sub.add_parser("mine", help="build a corpus (dev: text docs + byte tokenizer)")
    p_mine.add_argument("--config", default="configs/flint.yaml")
    p_mine.add_argument("--docs", help="text file, one document per line")
    p_mine.add_argument("--traces", help="text file, one agent trace per line")
    p_mine.add_argument("--out", default="runs/corpus")
    p_mine.add_argument("--n-c4", type=int, default=None)
    p_mine.add_argument("--n-xlong", type=int, default=None)

    p_train = sub.add_parser("train", help="run a training stage")
    p_train.add_argument("--config", default="configs/flint.yaml")
    p_train.add_argument("--stage", choices=["warmup", "distill"], default="warmup")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--steps", type=int, default=None)
    p_train.add_argument("--seq-len", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--corpus", default="synthetic",
                         help="synthetic | path to a .json token list from `mine`")
    p_train.add_argument("--corpus-size", type=int, default=200_000)
    p_train.add_argument("--resume", default=None, help="checkpoint path")
    p_train.add_argument("--out", default="runs/train")

    p_gate = sub.add_parser("gate", help="run the dtype-tier parity battery")
    p_gate.add_argument("--config", default="configs/flint.yaml")
    p_gate.add_argument("--device", default="cpu")
    p_gate.add_argument("--checkpoint", default=None)
    p_gate.add_argument("--out", default="runs/gate")

    p_export = sub.add_parser("export", help="export a portable checkpoint")
    p_export.add_argument("--config", default="configs/flint.yaml")
    p_export.add_argument("--checkpoint", default=None)
    p_export.add_argument("--device", default="cpu")
    p_export.add_argument("--out", default="runs/export")

    return parser


def _load_target_config(path: str) -> tuple[KSDraftConfig, dict, dict]:
    import yaml

    with open(path) as f:
        spec = yaml.safe_load(f)
    d = dict(spec.get("draft") or {})
    d.setdefault("vocab_size", int(spec["target"]["vocab_size"]))
    fields = {k.name for k in KSDraftConfig.__dataclass_fields__.values()} \
        if hasattr(KSDraftConfig, "__dataclass_fields__") else set()
    cfg = KSDraftConfig(**{k: v for k, v in d.items() if k in fields})
    return cfg, spec.get("teacher") or {}, spec.get("training") or {}


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


def _cmd_mine(args: argparse.Namespace) -> int:
    import json
    import random
    from pathlib import Path

    from .data import CorpusConfig, build_corpus, train_val_split

    if not args.docs:
        print("--docs is required for mining (dev corpus uses `train --corpus synthetic`)",
              file=sys.stderr)
        return 2
    _, _, tspec = _load_target_config(args.config)
    cspec = (tspec.get("corpus") or {})
    cfg = CorpusConfig(
        n_c4=args.n_c4 if args.n_c4 is not None else int(cspec.get("n_c4", 3500)),
        n_agent=int(cspec.get("n_agent", 800)),
        n_long=int(cspec.get("n_long", 48)),
        n_xlong=(args.n_xlong if args.n_xlong is not None
                 else int(cspec.get("n_xlong", 128))),
        val_frac=float(cspec.get("val_frac", 0.01)),
    )
    docs = Path(args.docs).read_text().splitlines()
    traces = (Path(args.traces).read_text().splitlines()
              if args.traces else ["<tool>noop</tool> ok"])
    # dev tokenizer: byte values (works for any vocab >= 256)
    tokenize = lambda s: [ord(c) % 256 for c in s]
    ids = build_corpus(docs, traces, tokenize, cfg, rng=random.Random(11))
    tr, va = train_val_split(ids, cfg.val_frac, random.Random(11))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_tokens.json").write_text(json.dumps(tr))
    (out / "val_tokens.json").write_text(json.dumps(va))
    print(f"mined {len(tr)} train / {len(va)} val tokens -> {out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import torch

    from .data import synthetic_ids
    from .model import KSDraftModel
    from .teacher import FixtureTeacher
    from .train import Trainer, TrainConfig, WindowSampler

    cfg, tspec, trspec = _load_target_config(args.config)
    if args.corpus == "synthetic":
        ids = synthetic_ids(args.corpus_size, cfg.vocab_size)
    else:
        ids = json.loads(Path(args.corpus).read_text())
    sampler = WindowSampler(ids, seq_len=args.seq_len or 64, seed=11)

    n_taps = len((tspec.get("state_taps") or [6, 14, 22, 30, 38, 48]))
    teacher = None
    teacher_dim = 0
    if args.stage == "distill":
        d_tap = int(tspec.get("fixture_d_tap", 64))
        teacher = FixtureTeacher(cfg.vocab_size, d_tap=d_tap, n_taps=n_taps,
                                 topk=int(tspec.get("topk_logits", 32)))
        teacher_dim = n_taps * d_tap

    torch.manual_seed(0)
    model = KSDraftModel(cfg, teacher_dim=teacher_dim)
    tcfg = TrainConfig(
        steps=args.steps or int(trspec.get("steps", 50)),
        seq_len=args.seq_len or 64,
        batch_size=args.batch_size or 4,
        peak_lr=float((trspec.get("optimizer") or {}).get("peak_lr", 3e-4)),
    )
    trainer = Trainer(model, teacher, sampler, tcfg, device=args.device,
                      out_dir=args.out)
    if args.resume:
        ck = trainer.load(args.resume)
        print(f"resumed from {args.resume} (step {ck['step']})")
    res = trainer.train(stage=args.stage)
    print(f"final loss {res['final']['loss']:.4f} (initial {res['initial']['loss']:.4f})")
    return 0


def _load_model(args) -> "object":
    """Build a model from the config; optionally overlay a checkpoint."""
    import torch

    from .model import KSDraftModel

    cfg, tspec, _ = _load_target_config(args.config)
    n_taps = len((tspec.get("state_taps") or [6, 14, 22, 30, 38, 48]))
    d_tap = int(tspec.get("fixture_d_tap", 64))
    model = KSDraftModel(cfg, teacher_dim=n_taps * d_tap)
    if getattr(args, "checkpoint", None):
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
    return model


def _cmd_gate(args: argparse.Namespace) -> int:
    from .gate import run_gate

    model = _load_model(args).to(args.device)
    report = run_gate(model=model, device=args.device, out_dir=args.out)
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}: {c['detail']}")
    print(f"gate {'PASSED' if report['all_pass'] else 'FAILED'} -> {args.out}/gate_report.json")
    return 0 if report["all_pass"] else 1


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export_model

    model = _load_model(args).to(args.device)
    path = export_model(model, args.out)
    print(f"exported -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "size":
        return _cmd_size(args)
    if args.command == "mine":
        return _cmd_mine(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "gate":
        return _cmd_gate(args)
    if args.command == "export":
        return _cmd_export(args)
    print(
        f"'{args.command}' is staged for a later milestone (see RESEARCH.md); "
        "not implemented yet.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
