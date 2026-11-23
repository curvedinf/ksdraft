# KSDraft

Parallel block-drafting draft models for
[speculative decoding](https://arxiv.org/abs/2211.17192).

Architecture:

- **DFly2-style target-KV reuse** (primary reference:
  [Tencent Hy4-flash DFly2](https://github.com/Tencent-Hunyuan/Hy4-flash)) —
  the draft's time-mixing layer is cross-attention into target-cached KV;
  no drafter-side KV cache
- **Untied borrowed tables** (DFly2 pattern): input = the target's
  embedding, head = the deployed unembed's dequantized grid; both
  frozen, both omitted from `--share-vocab` exports
- **KS3 decoders** with the fused fast path (GS32 activation
  grids, in-kernel int8 coefficient grids, NL Horner ladder, fused VJP) —
  [K-Splanifolds paper](https://zenodo.org/records/18673035),
  [ks3_int8 reference](https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks3_int8.py)
- **DSpine depth-carried causal injection** —
  [paper](https://arxiv.org/abs/2609.36173)
- **DFlash2-style block head** (offsets + dynamic conv + pairwise
  selector + last-write refinement) —
  [DFlash](https://arxiv.org/abs/2602.06036),
  [DFlash2 card](https://docs.nvidia.com/nemo/automodel/nemo_automodel/nemo_automodel/components/speculative/dflash/draft_qwen3_dflash2.md)
- **Split Muon + 8-bit AdamW** (block-wise int8 m/√v) —
  [Muon](https://kellerjordan.github.io/posts/muon/),
  [Muon scaling](https://arxiv.org/abs/2502.16982),
  [8-bit optimizers](https://arxiv.org/abs/2110.02861),
  [modded-nanogpt reference](https://github.com/KellerJordan/modded-nanogpt)
- **PTQR rounding-dither annealing** to the INT8 serving tier
  ([draft card](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR),
  [int8-vllm](https://github.com/curvedinf/int8-vllm))

Status: implementation in progress; no training runs yet. The full
architecture research, decisions, and audits live in
[RESEARCH.md](RESEARCH.md) — it is the source of truth for every design
choice in this repo (see Section 14 for the current target and shape).

## Target models

The draft attaches to any transformer target that exposes per-layer cached
K/V. A target adapter YAML carries the hidden size, KV width and head
count, vocab, sliding window, and the tensor names of the embedding and
unembed. The draft trains against the same int8 g128 KV grid the target
serves with — the exact-tier-student doctrine applied to the reuse seam.

## Draft shape

Configured per target (`configs/<target>.yaml`): `layers`, `d_model`,
`kv_dim`, `n_kv_heads`, `ks3_charts`, `block_len`. Example — 4 layers x
d=2304, 32Q/8KV GQA adapters (kv_dim 1024), KS3 24,800 charts — gives a
**trunk of 499,850,240 params (~500M excluding the borrowed tables)**.
`d_model` must match the target hidden so the embedding copy and the head
dequant stay exact.

## Block diagram

```
              +---------------------------------------------------+
              |  target model (frozen, external)                   |
              |  embedding [V,d]              unembed [V,d]        |
              +------+--------------------------------+---------+
                     | token ids                      | logits
                     v                                |
+-----------------------------------------------------------------------+
|  draft trunk                                                          |
|                                                                       |
|    x = LN_in(embed[tokens])                                           |
|                                                                       |
|    layer 0..N-1:                                                      |
|      target K/V tap (int8 g128 codes + scales, one per layer)         |
|               |                                                       |
|               v                                                       |
|      x --> LN --> TargetKVAttention --+--> h                          |
|               Q from the draft, K/V adapters from the tap             |
|      h --> LN --> KS3 (down -> Horner spine -> ReLU^2 -> up) --+--> y |
|      y --> DSpine AdjacentInjection (gate on the previous position)   |
|                                                                       |
|    h_out = LN_out(x)                                                  |
+-----------------------------------------------------------------------+
                                   | anchor = h_out[:, -1]
                                   v
                     +------------------------------+
                     | block head                   |
                     | hs_k = BlockConv(anchor +    |
                     |        block_delta_k)        |
                     | logits_k = hs_k @ unembed^T  |
                     | greedy walk over offsets     |
                     +------------------------------+
```

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
# shape budget (trunk-only; tables are borrowed at serve)
ksdraft size --layers 4 --d-model 2304 --kv-dim 1024 --ks3-charts 24800 --vocab 201024

# init the two borrowed tables from the target checkpoint
python scripts/init_embedding.py --copy --device cpu \
    --target <target-model-dir> --tensor model.embed_tokens.weight \
    --d-model 2304 --out runs/embedding_init.pt
python scripts/init_embedding.py --unembed \
    --target <target-model-dir> --out runs/head_init.pt

# smoke train (fixture KV — synthetic, dev only)
ksdraft train --config configs/<target>.yaml --stage warmup --device cuda \
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
configs/       # target adapters (one YAML per target)
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
