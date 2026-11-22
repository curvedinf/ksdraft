# KSDraft

Parallel block-drafting draft models for speculative decoding.

Architecture:

- **DFly2-style target-KV reuse** (primary reference:
  [Tencent Hy4-flash DFly2](https://github.com/Tencent-Hunyuan/Hy4-flash)) —
  the draft's time-mixing layer is cross-attention into target-cached KV;
  no drafter-side KV cache
- **Untied borrowed tables** (DFly2 pattern): input = the target's BF16
  embedding, head = the deployed W4 unembed's dequantized grid; both
  frozen, both omitted from `--share-vocab` exports
- **KS3 decoders** with the world-record fast path (GS32 activation
  grids, in-kernel int8 coefficient grids, NL Horner ladder, fused VJP)
- **DSpine depth-carried causal injection**
- **DFlash2-style block head** (offsets + dynamic conv + pairwise
  selector + last-write refinement)
- **Split Muon + 8-bit AdamW** (block-wise int8 m/√v, modded-nanogpt
  reference math)
- **PTQR rounding-dither annealing** to the INT8 serving tier

Status: implementation in progress; no training runs yet. The full
architecture research, decisions, and audits live in
[RESEARCH.md](RESEARCH.md) — it is the source of truth for every design
choice in this repo (see Section 14 for the current target and shape).

## Target

**Inkling-2-Mini** (26.4B/5.4B-effective looped MoE), served as the
finalized PTQR **W4A8-G128** checkpoint
(`~/models/Inkling-2-Mini-PTQR-W4A8-G128`, recipe in the
`../int8-vllm` fork: TP4 / fp16 / `int8_block_g128` KV / T=1.0). The
draft trains against the deployed KV grid and head grid — the
exact-tier-student doctrine applied to the reuse seam.

## Decided draft shape (`configs/inkling.yaml`)

4 layers x d=2304 (= target hidden), 32Q/8KV GQA adapters (kv_dim 1024),
KS3 24,800 charts: **trunk 499,850,240 (~500M excluding the borrowed
tables)**. d_model matching the target hidden is what makes the verbatim
embedding copy and the W4 head dequant exact.

## Install

Into a ROCm/CUDA torch environment:

```bash
pip install --no-build-isolation -e .
pip install --no-build-isolation -e ".[dev]"
```

## Running on the MI100 server

See [MI100.md](MI100.md) - environment recipe, memory budget, testing
ladder, stage commands, and known boundaries.

## Quick start

```bash
# decided shape budget (trunk-only; tables are borrowed at serve)
ksdraft size --layers 4 --d-model 2304 --kv-dim 1024 --ks3-charts 24800 --vocab 201024

# init the two borrowed tables from the finalized target
python scripts/init_embedding.py --copy --device cpu \
    --target ~/models/Inkling-2-Mini-PTQR-W4A8-G128 \
    --tensor model.llm.embed.weight --d-model 2304 --out runs/embedding_init.pt
python scripts/init_embedding.py --unembed \
    --target ~/models/Inkling-2-Mini-PTQR-W4A8-G128 --out runs/head_init.pt

# smoke train (fixture KV — synthetic, dev only)
ksdraft train --config configs/inkling.yaml --stage warmup --device cuda \
    --steps 8 --kv-fixture

# trunk-only serving artifact (the engine lends both tables)
ksdraft export --share-vocab ...
```

## Layout

```
ksdraft/
  config.py     # KSDraftConfig — shape + dtype tiers + exact budgets
  cli.py        # ksdraft size | mine | train | gate | export
  model/        # ReuseBlock + TargetKVAttention, KS3, DSpine, KSDraftModel
  kernels/      # ks3_wr.py (WR fast path), int8 tier, fused train losses
  teacher/      # teacher clients: fixture (KV!) | HF bf16 | HTTP logprobs
  train/        # staged trainer + AdamW8bit (split Muon default)
  data/         # corpus builders
configs/        # target adapters (flint.yaml historical; inkling.yaml current)
tests/
```

## Dtype tiers

- `int8` — intended target: W8A8 GS128, maximal fusion (training AND serving)
- `bf16` — generic fallback: same graph, portable kernels, parity reference
- `fp32` — debug/parity tier for the test battery

Training is INT8-native regardless of the serving tier.

## License

MIT — see [LICENSE](LICENSE). The vendored KS3 fast path
(`ksdraft/kernels/ks3_wr.py`) derives from k-splanifolds-lineage code
marked CC-BY-SA-4.0; provenance is recorded in the file.
