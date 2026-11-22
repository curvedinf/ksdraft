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
import os
import sys

from .config import KSDraftConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ksdraft",
        description="Parallel block-drafting draft models (RWKV-8 Rosa + KS3).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_size = sub.add_parser("size", help="print the draft-shape parameter budget")
    p_size.add_argument("--propose", action="store_true",
                        help="propose a shape from target specs (Section 9)")
    p_size.add_argument("--budget", type=float, default=1.0e9,
                        help="core parameter budget for --propose")
    p_size.add_argument("--target-active", type=float, default=2.5e9)
    p_size.add_argument("--target-total", type=float, default=39.8e9)
    p_size.add_argument("--layers", type=int, default=KSDraftConfig.layers)
    p_size.add_argument("--d-model", type=int, default=KSDraftConfig.d_model)
    p_size.add_argument("--head-size", type=int, default=KSDraftConfig.head_size)
    p_size.add_argument("--vocab", type=int, default=KSDraftConfig.vocab_size)
    p_size.add_argument("--rosa-streams", type=int, default=KSDraftConfig.rosa_streams)
    p_size.add_argument("--rosa-bits", type=int, default=KSDraftConfig.rosa_bits)
    p_size.add_argument("--ks3-charts", type=int, default=0, help="0 = density default (3.83 per width unit)")
    p_size.add_argument("--dtype", default="int8", choices=["int8", "bf16", "fp32"])
    p_size.add_argument("--kv-dim", type=int, default=0,
                        help=">0: DFly2-style target-KV reuse backbone")
    p_size.add_argument("--kv-window", type=int, default=1024)
    p_size.add_argument("--n-kv-heads", type=int, default=8)

    p_mine = sub.add_parser("mine", help="build a corpus (dev docs or --live)")
    p_mine.add_argument("--live", action="store_true",
                        help="regenerate the corpus with the served target")
    p_mine.add_argument("--model", default=None,
                        help="tokenizer/model dir for --live")
    p_mine.add_argument("--base-url", default="http://127.0.0.1:8030/v1")
    p_mine.add_argument("--api-key", default=None)
    p_mine.add_argument("--n-code", type=int, default=3000)
    p_mine.add_argument("--n-agent", type=int, default=300)
    p_mine.add_argument("--n-doc", type=int, default=100)
    p_mine.add_argument("--n-chat", type=int, default=400)
    p_mine.add_argument("--max-new", type=int, default=512)
    p_mine.add_argument("--concurrency", type=int, default=12)
    p_mine.add_argument("--config", default="configs/flint.yaml")
    p_mine.add_argument("--docs", help="text file, one document per line")
    p_mine.add_argument("--traces", help="text file, one agent trace per line")
    p_mine.add_argument("--out", default="runs/corpus")
    p_mine.add_argument("--n-c4", type=int, default=None)
    p_mine.add_argument("--n-xlong", type=int, default=None)

    p_train = sub.add_parser("train", help="run a training stage")
    p_train.add_argument("--config", default="configs/flint.yaml")
    p_train.add_argument("--stage", choices=["warmup", "distill", "foresight",
                                             "refine", "acceptance", "ptqr"],
                         default="warmup")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--steps", type=int, default=None)
    p_train.add_argument("--seq-len", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--corpus", default="synthetic",
                         help="synthetic | path to a .json token list from `mine`")
    p_train.add_argument("--corpus-size", type=int, default=200_000)
    p_train.add_argument("--resume", default=None, help="checkpoint path")
    p_train.add_argument("--out", default="runs/train")
    p_train.add_argument("--block-n", type=int, default=0,
                         help="parallel block-draft offsets to supervise")
    p_train.add_argument("--block-weight", type=float, default=None)
    p_train.add_argument("--block-only-steps", type=int, default=0)
    p_train.add_argument("--train-dtype", default=None,
                         choices=["fp16", "bf16", "fp32"])
    p_train.add_argument("--grad-accum", type=int, default=None)
    p_train.add_argument("--muon", action="store_true", default=None)
    p_train.add_argument("--no-muon", dest="muon", action="store_false",
                         default=None,
                         help="disable the split Muon arm (default: on)")
    p_train.add_argument("--no-adam8bit", dest="adam8bit",
                         action="store_false", default=None,
                         help="fp32 Adam m/v states (default: block-wise "
                              "int8 states with fp32 math)")
    p_train.add_argument("--muon-lr", type=float, default=None)
    p_train.add_argument("--ema-decay", type=float, default=None)
    p_train.add_argument("--lr-schedule", default=None,
                         choices=["cosine", "wsd"])
    p_train.add_argument("--block-position-decay", type=float, default=None)
    p_train.add_argument("--block-wtv", type=float, default=None,
                         help="block probability-L1 weight (DSpine Eq. 7)")
    p_train.add_argument("--lwr-weight", type=float, default=None,
                         help="DSpine last-write-refinement CE weight")
    p_train.add_argument("--selector-weight", type=float, default=None,
                         help="DFlash2 pairwise path-selector CE weight")
    p_train.add_argument("--block-ce-weight", type=float, default=None,
                         help="hard block-CE auxiliary in the distill stage")
    p_train.add_argument("--teacher-http", default=None,
                         help="served /v1 endpoint for live teacher signals")
    p_train.add_argument("--teacher-key", default=None)
    p_train.add_argument("--teacher-model", default=None)
    p_train.add_argument("--transfer-space", default=None,
                         help="DSpine transfer-space .pt (C, R_init, rank)")
    p_train.add_argument("--transfer-gamma", type=float, default=None)
    p_train.add_argument("--spine-sub-prob", type=float, default=None)
    p_train.add_argument("--embedding-init", default=None,
                         help=".pt tensor [vocab, d_model] (target SVD projection)")
    p_train.add_argument("--head-init", default=None,
                         help=".pt tensor [vocab, d_model] — the dequantized "
                              "W4 target unembed (untied-head recipe)")
    p_train.add_argument("--freeze-embeddings", action="store_true",
                         default=None)
    p_train.add_argument("--roll-every", type=float, default=None,
                         help="seconds between rolling ckpt_latest.pt saves")
    p_train.add_argument("--eval-cache", default=None,
                         help="teacher cache over held-out windows")
    p_train.add_argument("--eval-windows", type=int, default=None)
    p_train.add_argument("--boundary-sep", default=None,
                         help="comma-separated token ids marking a document "
                              "start (windows never cross one)")
    p_train.add_argument("--lr", type=float, default=None,
                         help="override the recipe peak learning rate")
    p_train.add_argument("--warmup-steps", type=int, default=None)
    p_train.add_argument("--save-every", type=int, default=None,
                         help="checkpoint interval in steps (default: steps/10)")
    p_train.add_argument("--kv-dim", type=int, default=None,
                         help=">0: DFly2-style target-KV reuse backbone "
                              "(overrides the config file)")
    p_train.add_argument("--kv-window", type=int, default=None)
    p_train.add_argument("--n-kv-heads", type=int, default=None)
    p_train.add_argument("--kv-fixture", action="store_true", default=False,
                         help="teacherless stages: deterministic synthetic KV "
                              "provider (smoke/dev only, not a real teacher)")
    p_train.add_argument("--kv-store", default=None,
                         help="mined signal store (scripts/"
                              "mine_inkling_signals.py): positioned KV "
                              "streams + teacher top-k for training")

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
    p_export.add_argument("--dtype", default="fp16",
                          choices=["fp16", "bf16", "fp32", "int8"],
                          help="artifact dtype (int8 stores codes+scales)")
    p_export.add_argument("--share-vocab", action="store_true", default=False,
                          help="omit the tied embedding table (DFly2 pattern: "
                               "the serving side binds the target's own)")

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


def _propose_shape(vocab: int, budget: float, layers: int = 24,
                   head: int = 64, kv_dim: int = 0,
                   n_kv_heads: int = 8) -> KSDraftConfig:
    """RESEARCH Section 9 proposal: widest head-aligned d under the budget.

    Depth stays at the family constant (24), streams track d/3, charts track
    d*368/96, and the tied embedding (vocab x d) is part of the budget.
    kv_dim > 0 sizes the DFly2-style reuse attention instead of Rosa.
    """
    best = None
    for d in range(head, 4097, head):
        s = max(1, round(d / 3))
        charts = max(1, round(d * 368 / 96))
        c = KSDraftConfig(vocab_size=vocab, layers=layers, d_model=d,
                          head_size=head, rosa_streams=s,
                          ks3_charts=charts, kv_dim=kv_dim,
                          n_kv_heads=n_kv_heads)
        if c.total_params() <= budget:
            best = c
        else:
            break
    if best is None:
        raise SystemExit("budget too small for the minimum shape")
    return best


def _cmd_size(args: argparse.Namespace) -> int:
    if getattr(args, "propose", False):
        cfg = _propose_shape(args.vocab, args.budget, layers=args.layers,
                             head=args.head_size,
                             kv_dim=getattr(args, "kv_dim", 0) or 0,
                             n_kv_heads=getattr(args, "n_kv_heads", 8) or 8)
        ratio = cfg.total_params() / max(1.0, args.target_active)
        mixer = "kv-reuse" if cfg.kv_dim else "rosa"
        print(f"proposed shape for vocab={args.vocab:,}, "
              f"budget={args.budget/1e9:.2f}B core ({mixer} mixer)")
        print(f"  d_model {cfg.d_model}, layers {cfg.layers}, "
              f"streams {cfg.rosa_streams}, charts {cfg.charts}")
        if cfg.kv_dim:
            print(f"  attention {cfg.q_heads} q-heads <- {cfg.n_kv_eff} "
                  f"kv-heads (kv_dim {cfg.kv_dim}, W {cfg.kv_window})")
        print(f"  core {cfg.total_params():,} "
              f"({cfg.total_params()/1e9:.3f}B)")
        if args.target_active:
            print(f"  draft/target active FLOP ratio: {ratio:.2f} "
                  f"(Section 9 window ~0.2-0.5; 1B/2.5B = 0.40)")
        print(f"  draft size vs target total: "
              f"{cfg.total_params()/max(1.0, args.target_total)*100:.1f}%")
        return 0
    cfg = KSDraftConfig(
        vocab_size=args.vocab,
        layers=args.layers,
        d_model=args.d_model,
        head_size=args.head_size,
        rosa_streams=args.rosa_streams,
        rosa_bits=args.rosa_bits,
        ks3_charts=args.ks3_charts,
        dtype=args.dtype,
        kv_dim=getattr(args, "kv_dim", 0) or 0,
        n_kv_heads=getattr(args, "n_kv_heads", 8) or 8,
        kv_window=getattr(args, "kv_window", 1024) or 1024,
    )
    s = cfg.summary()
    print(f"ksdraft shape budget (dtype={s['dtype']})")
    print(f"  layers            {s['layers']}")
    print(f"  d_model           {s['d_model']}  (head granularity: {s['n_heads']} x {s['head_size']})")
    if s["kv_dim"]:
        print(f"  attn(kv-reuse)    {s['attn_params_per_layer']:>15,}  "
              f"({s['q_heads']} q-heads <- {s['n_kv_heads']} kv-heads, "
              f"kv_dim {s['kv_dim']}, W {s['kv_window']})")
    else:
        print(f"  rosa              {s['rosa_streams']} streams x {s['rosa_bits']} bit(s)")
        print(f"  rosa/layer        {s['rosa_params_per_layer']:>15,}")
    print(f"  ks3 charts        {s['ks3_charts']}")
    print(f"  ks3/layer         {s['ks3_params_per_layer']:>15,}")
    print(f"  total/layer       {s['layer_params']:>15,}")
    print(f"  embedding (tied)  {s['embedding_params']:>15,}")
    print(f"  TOTAL             {s['total_params']:>15,}  (~{s['total_params'] / 1e9:.2f}B)")
    return 0


def _cmd_mine(args: argparse.Namespace) -> int:
    import importlib.util
    import json
    import random
    from pathlib import Path

    if getattr(args, "live", False):
        if not args.model:
            print("mine --live requires --model (the served model dir)",
                  file=sys.stderr)
            return 2
        script = (Path(__file__).resolve().parents[1]
                  / "scripts/build_onpolicy_corpus.py")
        spec = importlib.util.spec_from_file_location(
            "_ksdraft_onpolicy", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ns = argparse.Namespace(
            model=args.model, base_url=args.base_url,
            api_key=(args.api_key or os.environ.get("VLLM_API_KEY", "")),
            corpus=args.corpus if getattr(args, "corpus", None) else
            "/home/curved/SDGraft/data/flint_longctx/train_tokens.pt",
            out=args.out, n_code=args.n_code, n_agent=args.n_agent,
            n_doc=args.n_doc, n_chat=args.n_chat, max_new=args.max_new,
            doc_chars=8000, doc_new=256, max_prompt=12000,
            concurrency=args.concurrency, greedy_frac=0.2, seed=11)
        return mod.run(ns)

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
    if getattr(args, "kv_dim", None) is not None:
        cfg.kv_dim = args.kv_dim
    if getattr(args, "kv_window", None) is not None:
        cfg.kv_window = args.kv_window
    if getattr(args, "n_kv_heads", None) is not None:
        cfg.n_kv_heads = args.n_kv_heads
    bounds = None
    if args.corpus == "synthetic":
        ids = synthetic_ids(args.corpus_size, cfg.vocab_size)
    elif str(args.corpus).endswith(".pt"):
        # Pre-tokenized token stream (e.g. the PTQR 512k corpus): keep it as a
        # tensor so multi-million-token corpora do not become Python lists.
        ids = torch.load(args.corpus, map_location="cpu", weights_only=True)
        if isinstance(ids, dict):
            bounds = ids.get("boundaries")
            ids = ids.get("tokens", ids.get("ids"))
        if bounds is None:
            sib = Path(str(args.corpus)).with_name("boundaries.pt")
            if sib.exists():
                bounds = torch.load(sib, map_location="cpu",
                                    weights_only=True)
    else:
        ids = json.loads(Path(args.corpus).read_text())
    if args.boundary_sep:
        from .train import detect_boundaries
        sep = [int(t) for t in str(args.boundary_sep).split(",") if t != ""]
        bounds = detect_boundaries(
            ids if isinstance(ids, torch.Tensor) else torch.tensor(ids), sep)
        print(f"boundaries: {bounds.numel()} documents", flush=True)
    mined = None
    if getattr(args, "kv_store", None):
        from .teacher import MinedSignals
        mined = MinedSignals(
            args.kv_store,
            window=int(getattr(args, "kv_window", None) or cfg.kv_window))
        print(f"kv: mined store {args.kv_store} "
              f"(W={mined.window}, positioned)", flush=True)
        # sampling must stay inside the store's coverage
        n_cov = int(mined.k.shape[0])
        if len(ids) > n_cov:
            ids = ids[:n_cov]
            if bounds is not None:
                bounds = bounds[bounds <= n_cov]
    sampler = WindowSampler(ids, seq_len=args.seq_len or 64, seed=11,
                            boundaries=bounds)

    n_taps = len((tspec.get("state_taps") or [6, 14, 22, 30, 38, 48]))
    teacher = None
    teacher_dim = 0
    if getattr(args, "teacher_http", None):
        # Live logits teacher over the served recipe (Stages 2/5/6).
        from .teacher.http import HTTPTeacher
        teacher = HTTPTeacher(
            args.teacher_http,
            model=getattr(args, "teacher_model", None) or "flint",
            api_key=(getattr(args, "teacher_key", None)
                     or os.environ.get("VLLM_API_KEY", "")),
            topk=int(tspec.get("topk_logits", 32)))
        print(f"teacher: HTTP {args.teacher_http} ({teacher.model})", flush=True)
    elif args.stage == "distill":
        d_tap = int(tspec.get("fixture_d_tap", 64))
        teacher = FixtureTeacher(cfg.vocab_size, d_tap=d_tap, n_taps=n_taps,
                                 topk=int(tspec.get("topk_logits", 32)),
                                 kv_dim=cfg.kv_dim,
                                 kv_window=min(cfg.kv_window,
                                               args.seq_len or 64),
                                 kv_taps=cfg.layers if cfg.kv_dim else None)
        teacher_dim = n_taps * d_tap

    torch.manual_seed(0)
    emb_init = None
    emb_path = (getattr(args, "embedding_init", None)
                or trspec.get("embedding_init"))
    if emb_path:
        emb_init = torch.load(emb_path, map_location="cpu",
                              weights_only=True)
    head_init = None
    head_path = (getattr(args, "head_init", None)
                 or trspec.get("head_init"))
    if head_path:
        head_init = torch.load(head_path, map_location="cpu",
                               weights_only=True)
    ts_space = None
    if getattr(args, "transfer_space", None):
        ts_space = torch.load(args.transfer_space, map_location="cpu",
                              weights_only=True)
    model = KSDraftModel(cfg, teacher_dim=teacher_dim, embedding_init=emb_init,
                         head_init=head_init, transfer_space=ts_space)
    if str(cfg.dtype).lower() == "int8":
        from .model import to_int8
        model = to_int8(model, trainable_ks3=True)
    kv_provider = None
    if mined is not None:
        kv_provider = mined
    elif cfg.kv_dim:
        print(f"reuse backbone: kv_dim={cfg.kv_dim} "
              f"({cfg.q_heads} q-heads <- {cfg.n_kv_eff} kv-heads, "
              f"W={cfg.kv_window})", flush=True)
        if teacher is None:
            if getattr(args, "kv_fixture", False):
                from .teacher import KVFixture
                kv_provider = KVFixture(
                    cfg.layers, cfg.kv_dim,
                    kv_window=min(cfg.kv_window, args.seq_len or 64), seed=11)
                print("kv: fixture provider (synthetic — smoke/dev only)",
                      flush=True)
            else:
                raise SystemExit(
                    "kv_dim>0 (reuse backbone) needs a KV source for "
                    "teacherless stages: --kv-fixture (synthetic smoke/dev) "
                    "or a teacher client whose signals() returns kv")
    tcfg = TrainConfig(
        steps=args.steps or int(trspec.get("steps", 50)),
        seq_len=args.seq_len or 64,
        batch_size=args.batch_size or 4,
        peak_lr=(args.lr if args.lr is not None
                 else float((trspec.get("optimizer") or {}).get("peak_lr", 3e-4))),
        warmup=(args.warmup_steps if args.warmup_steps is not None
                else TrainConfig.warmup),
        val_every=args.save_every or max(1, (args.steps or 50) // 10),
        block_n=args.block_n,
        block_weight=(args.block_weight if args.block_weight is not None
                      else TrainConfig.block_weight),
        block_only_steps=args.block_only_steps,
        train_dtype=(args.train_dtype if args.train_dtype is not None
                     else TrainConfig.train_dtype),
        grad_accum=(args.grad_accum if args.grad_accum is not None
                    else TrainConfig.grad_accum),
        muon=(bool(args.muon) if args.muon is not None
              else TrainConfig.muon),
        adam8bit=(bool(args.adam8bit) if args.adam8bit is not None
                  else TrainConfig.adam8bit),
        muon_lr=(args.muon_lr if args.muon_lr is not None
                 else TrainConfig.muon_lr),
        ema_decay=(args.ema_decay if args.ema_decay is not None
                   else TrainConfig.ema_decay),
        lr_schedule=(args.lr_schedule if args.lr_schedule is not None
                     else TrainConfig.lr_schedule),
        block_position_decay=(args.block_position_decay
                              if args.block_position_decay is not None
                              else TrainConfig.block_position_decay),
        freeze_embeddings=(bool(args.freeze_embeddings)
                           if args.freeze_embeddings is not None
                           else bool(trspec.get(
                               "freeze_embeddings",
                               TrainConfig.freeze_embeddings))),
        block_ce_weight=(args.block_ce_weight
                         if args.block_ce_weight is not None
                         else TrainConfig.block_ce_weight),
        selector_weight=(args.selector_weight
                         if args.selector_weight is not None
                         else TrainConfig.selector_weight),
        lwr_weight=(args.lwr_weight if args.lwr_weight is not None
                    else TrainConfig.lwr_weight),
        block_wtv_weight=(args.block_wtv if args.block_wtv is not None
                          else TrainConfig.block_wtv_weight),
        roll_every_sec=(args.roll_every if args.roll_every is not None
                        else TrainConfig.roll_every_sec),
        eval_cache=(args.eval_cache if args.eval_cache is not None
                    else TrainConfig.eval_cache),
        eval_windows=(args.eval_windows if args.eval_windows is not None
                      else TrainConfig.eval_windows),
        transfer_space=(args.transfer_space if args.transfer_space
                        else TrainConfig.transfer_space),
        transfer_gamma=(args.transfer_gamma if args.transfer_gamma is not None
                        else TrainConfig.transfer_gamma),
        spine_sub_prob=(args.spine_sub_prob
                        if args.spine_sub_prob is not None
                        else TrainConfig.spine_sub_prob),
    )
    trainer = Trainer(model, teacher, sampler, tcfg, device=args.device,
                      out_dir=args.out, kv_provider=kv_provider)
    if args.resume:
        ck = trainer.load(args.resume)
        print(f"resumed from {args.resume} (step {ck['step']})")
        del ck
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
    state = None
    tier = None
    if getattr(args, "checkpoint", None):
        sd = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        state = sd["model"] if "model" in sd else sd
        tier = sd.get("tier") if isinstance(sd, dict) else None
    # Teacher conditioning is a property of the checkpoint, not the recipe:
    # drafts trained with logits-only teachers have no teacher_proj.
    teacher_dim = (n_taps * d_tap
                   if state is not None
                   and any(k.startswith("teacher_proj") for k in state)
                   else 0)
    ts_space = None
    if state is not None and "spine_core.A.weight" in state:
        ts_space = {"rank": int(state["spine_core.A.weight"].shape[0])}
    model = KSDraftModel(cfg, teacher_dim=teacher_dim,
                         transfer_space=ts_space)
    if state is not None:
        # A PTQR checkpoint holds the converted module tree (KS3Int8Cell,
        # Int8Linear); the plain model's keys do not match it.
        if tier == "int8" or "blocks.0.ffn.down_codes" in state:
            from .model import to_int8
            has_masters = "blocks.0.ffn.master_up" in state
            model = to_int8(model, trainable_ks3=has_masters)
            model.load_state_dict(state, strict=True)
        else:
            missing, _ = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[load] {len(missing)} new parameter(s) fresh: {missing[:3]}")
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
    path = export_model(model, args.out, dtype=args.dtype,
                        share_vocab=bool(getattr(args, "share_vocab", False)))
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
