# RESEARCH.md — ksdraft technology survey

Reference notes for the draft-model architecture: a parallel block-drafting backbone
(RWKV-8 Rosa mixer + KS3 spine decoders), intra-block causal correction (DSpine),
adaptive scheduling (EAGLE-Harness), and draft/verification pipelining via JetSpec++
foresight drafting (decided 2026-11-08; D3D considered, not selected).
Training numerics decided 2026-11-08: full native INT8 forward and backward, with
FP16 gradient handoff to the optimizer (Section 8). INT8 checkpoint production via
PTQR from int8-vllm (Section 8.1).
Target model decided 2026-11-08: Recursive Flint (40B total / ~2.5B active MoE).
Draft shape decided 2026-11-08, corrected same day to the RWKV-8-only backbone:
~0.98B params — 24 layers, d=1792, head_size 64, 597 one-bit Rosa streams,
6,869 rank-1 KS3 charts, tied embeddings (Section 9).

Researched 2026-11-08. No implementation has begun.

---

## 1. Backbone / parallel block drafting

### DFlash — Block Diffusion for Flash Speculative Decoding
- **What:** Lightweight block-diffusion drafter proposes an entire block of tokens in a
  single forward pass under a block-causal schedule, conditioned on target-model
  context features (hidden states / KV). Verified losslessly by one target forward.
- **Results:** >6x lossless acceleration; up to 2.5x over prior SOTA. Deployed block
  sizes: 8 (NVIDIA Model Optimizer default), 16 (AngelSpec default).
- **Known weakness:** assumes intra-block conditional independence — parallel
  prediction ignores contextual dependencies between adjacent draft positions
  (motivates DCOCO, DSpine, LiLiCorr below).
- **Links:**
  - Paper: https://arxiv.org/abs/2602.06036
  - PMLR: https://proceedings.mlr.press/v306/chen26bv.html
  - NVIDIA Model Optimizer example (params, anchors, Qwen3-based components):
    https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/doc/dflash.md
  - AngelSpec concept docs (block-parallel drafter family):
    https://angelspec.readthedocs.io/en/latest/concepts/dflash.html
    https://angelspec.readthedocs.io/en/latest/concepts/draft_model_family.html

### DFlash2 (successor)
- **What:** DFlash stack plus in-block convs and a pairwise path selector that walks
  per-position candidates to pick a coherent path before verification — a built-in
  DCOCO-like corrector.
- **Links:**
  - NeMo AutoModel draft card:
    https://docs.nvidia.com/nemo/automodel/nemo_automodel/nemo_automodel/components/speculative/dflash/draft_qwen3_dflash2.md
  - MLX package: https://pypi.org/project/dflash-mlx/

### H-Spec — Parallel Speculative Decoding Without a Drafter-Side KV Cache
- **What:** Hybrid Mamba-attention parallel drafter. Mamba modules seeded with
  projected last-token target hidden states; attention modules reuse target KVs in
  place. No drafter-side KV cache. Target model unchanged.
- **Results:** Higher throughput and lower KV-cache utilization than baselines across
  concurrency levels; context-length robust.
- **Relevance:** evidence that a state-recurrent (Mamba/RWKV-class) parallel drafter is
  the right backbone family; informs how target context feeds the Rosa backbone.
- **Links:**
  - Paper: https://arxiv.org/abs/2609.24197 (HTML: https://arxiv.org/html/2609.24197v1)
  - SGLang integration PR (mamba_attn_hybrid, incl. NPU support):
    https://github.com/sgl-project/sglang/pull/42021

### Related family mentions
- **DSpark** — block-parallel drafter with dynamic verification logic (referenced by
  EAGLE-4 and JetSpec++ comparisons). See AngelSpec drafter-family docs above.
- **LiLiCorr** — lightweight likelihood correlation of parallel drafts; +9–19%
  acceptance length over vanilla DFlash drafter. Paper:
  https://browse-export.arxiv.org/pdf/2608.20530
  (Served in int8-vllm as the Nemotron Bolt draft — see Section 8.1.)
- **Block glancing for flash speculative decoding** — analysis of DFlash's
  conditional-independence assumption: https://arxiv.org/abs/2610.03712 (ScienceDirect:
  https://www.sciencedirect.com/science/article/abs/pii/S092523122602761X)

---

## 2. Intra-block correction (coherence)

### DSpine — Draft in Parallel, Condition Through Depth
- **What:** Parallel drafter with causal conditioning *inside* the backbone. At every
  layer, gated adjacent injection writes each predecessor's predicted feature into its
  successor (gate g = sigma(G_q z_t + G_p z_{t-1} + b) gating m = A z_{t-1}), so the
  causal chain unfolds over network depth while all positions still update in one
  parallel pass. Predicted features at every layer are supervised against token
  embeddings in a unified transfer space; predecessor-conditioned decoding at the head.
- **Results:** Longest acceptance length on all benchmarks (both models, both
  temperatures); round time ~7.14 ms, on par with DFlash2's 7.19 ms; ~10.9% lower
  decode time per token; highest-throughput speculative method at every task/concurrency.
- **Chosen role:** primary intra-block correction mechanism (integrated through depth,
  not a post-hoc head).
- **Links:**
  - Paper: https://arxiv.org/abs/2609.36173 (HTML v2: https://arxiv.org/html/2609.36173v2)
  - HF paper page: https://huggingface.co/papers/2609.36173
  - Model (Qwen3-8B-DSpine-b16, block 16): https://huggingface.co/tencent/Qwen3-8B-DSpine-b16

### DCOCO — Contextual Correction of Parallel Drafts
- **What:** "DFlash Contextual Corrector" — lightweight post-hoc corrector. Keeps
  top-k candidates per drafted position (top-16) and scores contextual compatibility
  across adjacent positions in one small batched pass (parallel, unlike sequential
  correctors). Uses information from a single drafter pass.
- **Results:** +6–23% over strongest competitor at 4–32 concurrent requests; 6.6x–19x
  throughput over DFlash from 1–32 streams (greedy); sequential correctors fall below
  uncorrected DFlash under load.
- **Chosen role:** fallback / cheap add-on if DSpine-style injection does not transfer
  to the Rosa backbone; also the model for candidate retention feeding the scheduler.
- **Links:**
  - Paper: https://arxiv.org/abs/2610.12570 (HTML: https://arxiv.org/html/2610.12570v1)
  - Code: https://github.com/two-delta/DCOCO
  - Model (Qwen3-8B-DCOCO drafter): https://huggingface.co/TD-Labs/Qwen3-8B-DCOCO
  - HF paper page: https://huggingface.co/papers/2610.12570

---

## 3. Adaptive scheduling (efficiency)

### EAGLE-4 / EAGLE-Harness
- **What:** EAGLE-4 drafter + `eagle-harness` controller. Warmup calibration measures
  real draft/target costs on the actual hardware/batch; per round it decides:
  (a) whether to draft one level deeper (depth-rising confidence threshold),
  (b) how many drafted tokens to send for verification — each additional token only
  while worth its measured verification cost, (c) tree vs chain strategy from load
  (dynamic tree <= 4 concurrent requests, dynamic chain from 8).
- **Results:** Adapts to batch size / hardware / workload; plugins for SGLang, vLLM,
  TokenSpeed.
- **Chosen role:** control loop for the drafting schedule — generalized from tree
  depth/width to block length, candidate fan-out, and stop decisions.
- **Links:**
  - Official repo: https://github.com/eagle-hq/EAGLE-4
  - Architecture doc: https://github.com/eagle-hq/EAGLE-4/blob/main/docs/architecture.md
  - Paper PDF: https://github.com/eagle-hq/EAGLE-4/blob/main/EAGLE4_Paper.pdf
  - PyPI: https://pypi.org/project/eagle-harness/
  - Design discussion (former EAGLE team, incl. DSpark comparison):
    https://qingkeai.online/blog/EAGLE
  - TokenSpeed integration PR: https://github.com/lightseekorg/tokenspeed/pull/2547
  - News summaries: https://aipolix.com/en-us/news/eagle-4-cost-aware-speculative-decoding-drafter ,
    https://lightseek.org/blog/eagle-4.html

---

## 4. Pipelining (throughput) — DECIDED: JetSpec++

**Decision (2026-11-08):** pipelining via JetSpec++ foresight drafting. The drafter
therefore *does* consume target hidden states (DFlash-style target-context
conditioning stays), which makes the foresight adapter — predicting target hidden
states and anchor tokens for anticipated verification outcomes — a required component
for overlapping drafting of block t+1 with verification of block t. D3D's
target-state-free route was considered and not selected.

### JetSpec++ — Foresight Parallel Drafting with Self Refinements
- **What:** Three components: (1) intra-block self refinement — drafter iteratively
  corrects its own parallel predictions within a block; (2) inter-block foresight
  drafting — an adapter anticipates verification outcomes and predicts the
  corresponding target hidden states and anchor tokens, so drafting of block t+1
  begins before block t is verified; (3) pipelined drafting/verification — overlap
  drives drafting's marginal wall-clock cost toward zero.
- **Results:** Up to 7.45x wall-clock over AR decoding, 1.54x over DSpark; mean
  acceptance length 8.94 across seven benchmarks; lossless.
- **Chosen role:** pipelining mechanism for ksdraft. Consequences for the design:
  - Backbone keeps target-hidden-state conditioning (dual-source context, DFlash /
    H-Spec style) rather than running fully decoupled.
  - The foresight adapter (target-state + anchor-token prediction) is a required
    trained component.
  - Intra-block self refinement is a natural complement to DSpine injection: refine
    the Rosa sweep's own draft in time freed by pipelining.
- **Links:**
  - Paper: https://arxiv.org/abs/2610.19792 (HTML v1: https://arxiv.org/html/2610.19792v1 ,
    v2: https://arxiv.org/html/2610.19792v2)
  - Code: https://github.com/hao-ai-lab/jetspec

### D3D — Decoupled Diffusion Drafting for Asynchronous Speculative Decoding
- **What:** Standalone diffusion drafter that removes the target-hidden-state
  dependency entirely; combined blockwise denoising with autoregressive auxiliary
  supervision. Independence lets it draft successor blocks during verification in one
  packed pass — asynchronous speculative decoding without foresight machinery.
- **Results:** High speedup; layer-prunable (36 -> 31 layers retains >99% of mean
  acceptance length, best tested speedup).
- **Status:** not selected (JetSpec++ chosen). Retained as reference for its async
  packing trick (successor blocks drafted during verification in one packed pass) and
  layer-pruning findings.
- **Links:**
  - Paper: https://arxiv.org/abs/2611.06425
  - HF paper page: https://huggingface.co/papers/2611.06425

### Related
- **SSD (speculative speculative decoding)** — draft on separate hardware,
  pre-computing during verification: https://www.linkedin.com/posts/emi-andere_in-speculative-decoding-the-draft-sits-idle-activity-7435768371352129536-x0zr
- **vLLM async scheduling x spec decoding compat PR #24799** (overlap infrastructure):
  https://github.com/vllm-project/vllm/pull/24799

---

## 5. Draft backbone components

### RWKV-8 Rosa / QKV-ROSA — mixer layers
- **What:** Hard retrieval mixer from Bo Peng (BlinkDL) RWKV-v8 examples
  (`samx_qkv_slow` / `ROSA_QKV_B_1bit` in `251024_rosaQKV_run.py`). n-bit discrete
  Q/K/V symbols (1–64 packed); each output takes the payload after one causal
  longest-suffix match: find the longest query suffix in strictly previous keys, take
  the latest matching endpoint, read that endpoint's successor value (+payload on
  matched 1, −payload on matched 0 / no match). Learned empty payload. Constant state,
  no KV cache.
- **Training contribution (the repo's purpose):** fast, exact single-bit
  counterfactual backpropagation — finite output change per Q/K/V bit flip projected
  on the incoming gradient. C++17 engine: O~(T^7/4) time, O~(T^4/3) space; first- or
  second-order credit; randomized verified cache (bitwise-identical); opt-in
  approximate mode. Bitwise-validated against brute force. MIT license.
- **Caveats:** CPU implementation currently (CUDA tensors require transfers);
  small-scale hybrid results (k-splanifolds, context 64) did not yet beat baseline.
- **Links:**
  - Repo: https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward
  - Upstream architecture: RWKV-LM / BlinkDL RWKV-v8 examples —
    https://github.com/BlinkDL/RWKV-LM

### KS3 — decoder (channel-mix / FFN) layers
- **What:** "Spine" FFN from the K-Splanifolds paper: learned input/output projections
  into chart subspaces (arbitrary count/rank), bounded Hermite cubic plus two warped
  Chebyshev terms (6 learned coefficients per output coordinate, warp scale/shift per
  chart), coordinate bypass, ReLU². Param count: `num_subspaces * (subspace_rank *
  (in + out + 6) + 2)`; ~73.6K per FFN at width 96 / 368 rank-1 subspaces.
- **`ks3_int8.py` (the linked implementation):** W8A8 with GS32 grouping and FP16
  scales; fused Triton forward (Horner geometry, fused input-GEMM/geometry at rank 1);
  optional INT8 tanh-LUT nonlinear mode; fused straight-through-estimator backward for
  QAT (FP32 masters, dequant-in-kernel FP32 backward GEMMs); tensor parallelism over
  whole charts; full-RWKV `convert()` helper keeping recurrent state FP32; Adam fusion
  and CUDA-graph capture helpers.
- **Evidence:** k-splanifolds `experiments/rwkv7_ks2` validated the integration shape —
  "RWKV-8-style RWKV7 + ROSA hybrid": RWKV delta-rule and ROSA computed from the same
  96D main residual (separate LayerNorms, summed) ahead of unchanged KS decoders,
  2048D decoder-only side residual, demuxed history; Rosa trained with exact
  counterfactual credits (max gradient error 4.8e-7 vs brute force). Honest caveat:
  at 15K steps / context 64, neither ROSA hybrid arm beat the all-MLP baseline; KS3
  FFNs slightly behind matched MLPs at small scale (case here is speed + fused int8).
- **Links:**
  - Repo: https://github.com/curvedinf/k-splanifolds
  - KS3 experiment dir (RWKV7 controller/FFN ablations, RWKV8 hybrid summary):
    https://github.com/curvedinf/k-splanifolds/tree/main/experiments/rwkv7_ks2
  - KS3 int8 kernels: https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks3_int8.py
  - KS3 PyTorch reference: https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks3.py
  - Diagrams notebook (KS1/KS2/KS3 geometry):
    https://github.com/curvedinf/k-splanifolds/blob/main/docs/diagrams/ks-blocks.ipynb
  - INT8/TP measurements report:
    https://github.com/curvedinf/k-splanifolds/blob/main/experiments/rwkv7_ks2/reports/2026-10-23-rank1-int8-tp/REPORT.md
  - Paper (Zenodo DOI): https://doi.org/10.5281/zenodo.18673034

---

## 6. Architecture map (which tech fills which role)

| Role | Technology | Primary links |
| --- | --- | --- |
| Target model (decided) | Recursive Flint — 40B total / ~2.5B active MoE coding agent (Section 9) | [HF](https://huggingface.co/RecursiveSI/flint) |
| Parallel block proposal | DFlash pattern on RWKV-8 Rosa mixer + KS3 decoders | [DFlash](https://arxiv.org/abs/2602.06036), [Rosa repo](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward), [KS3](https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks3_int8.py) |
| Draft shape (decided) | ~0.98B: 24 layers x d=1792, 597 one-bit Rosa streams, 6,869 KS3 charts, tied embeddings (Section 9) | [v6-Finch-1B6 config](https://huggingface.co/RWKV/v6-Finch-1B6-HF), [RWKV-7 paper](https://openreview.net/pdf/74011cffc1082ac9394afde92802c810c51baabf.pdf) |
| Recurrent drafter evidence | H-Spec (hybrid Mamba-attention, no drafter KV) | [arXiv](https://arxiv.org/abs/2609.24197) |
| Intra-block correction (primary) | DSpine gated adjacent injection, per-layer transfer-space supervision | [arXiv](https://arxiv.org/abs/2609.36173) |
| Intra-block correction (fallback) | DCOCO parallel contextual scorer, top-k candidates | [arXiv](https://arxiv.org/abs/2610.12570), [code](https://github.com/two-delta/DCOCO) |
| Adaptive scheduling | EAGLE-Harness cost/confidence controller | [repo](https://github.com/eagle-hq/EAGLE-4), [PyPI](https://pypi.org/project/eagle-harness/) |
| Pipelining (decided: JetSpec++) | JetSpec++ foresight drafting — drafter consumes target hidden states; foresight adapter predicts them ahead of verification; intra-block self refinement complements DSpine | [arXiv](https://arxiv.org/abs/2610.19792), [code](https://github.com/hao-ai-lab/jetspec) |
| Pipelining (not selected) | D3D decoupled drafter (no target-state dependency) | [arXiv](https://arxiv.org/abs/2611.06425) |
| Training numerics (decided) | Full native INT8 forward + INT8 backward, FP16 gradient handoff (Section 8) | [ks3_int8](https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks3_int8.py), [denoising QAT](https://github.com/curvedinf/k-splanifolds/tree/main/experiments/denoising) |
| INT8 checkpoint production + serving stack | PTQR rounding-dither retraining; int8-vllm W8A8 GS128 serving fork (Section 8.1) | [int8-vllm](https://github.com/curvedinf/int8-vllm), [PTQR draft card](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR) |
| Draft sizing (decided) | ~1B params, depth-heavy, gated by harness cost model vs Flint verify (Section 9) | [Flint-DFlash2 baseline](https://huggingface.co/incoai/Flint-DFlash2) |

## 7. Open design questions

- Rosa counterfactual backward is CPU-only; a CUDA port (or approximate mode) is a
  project pillar. Small-scale results do not yet establish Rosa drafting quality at
  real model sizes or long context.
- DSpine's injection was designed for continuous attention features; interaction with
  Rosa's discrete symbol quantization (inject before/after binarization) is novel.
- Resolved 2026-11-08: pipelining via JetSpec++ foresight drafting (not D3D).
  Consequences: the drafter consumes target hidden states, needs the trained
  foresight adapter (target-state + anchor-token prediction) to overlap drafting with
  verification, and keeps DFlash/H-Spec-style target-context injection; JetSpec++'s
  intra-block self refinement can refine the Rosa sweep's own drafts in freed time.
- KS3 trails matched MLPs slightly at small scale; its justification here is fused
  W8A8 speed, which is what a drafter needs.

---

## 8. Training numerics — DECIDED: full native INT8 forward and backward, FP16 gradient handoff

**Decision (2026-11-08):** ksdraft trains as a fully INT8 system. INT8 arithmetic in
the forward pass **and** throughout the backward pass (INT8 operands, INT32
accumulation in backward GEMMs/VJPs). Gradients are materialized as **FP16** — not
INT8 — only at the handoff to the optimizer. An earlier INT8-gradient torch backward
by the user exists elsewhere and will be remade rather than recovered.

**What the reference repo actually demonstrates (verified 2026-11-08):** INT8-forward
training is demonstrated; INT8 backward is not — every backward in k-splanifolds is
FP32 math:

| Implementation | INT8 in forward (training)? | Backward arithmetic |
| --- | --- | --- |
| [`ks3_int8.py`](https://github.com/curvedinf/k-splanifolds/blob/main/experiments/common/ks3_int8.py) (GPU Triton, GS32 W8A8, fused STE) | Yes — QAT with FP32 masters | FP32: "backward products and accumulators are FP32, not INT8" (dequantized operands in-kernel) |
| [`ks2_cpu`](https://github.com/curvedinf/k-splanifolds/tree/main/experiments/common/ks2_cpu) (CPU, GS128 INT8 VNNI) | No — frozen inference export | n/a (its fused training backward is analytic FP32 of the stabilized function) |
| [`denoising` TQ1 QAT](https://github.com/curvedinf/k-splanifolds/tree/main/experiments/denoising) (GS128 grouping) | Yes — all-weight ternary fake-quant | FP32: "QAT does not quantize optimizer state or backpropagation" |
| [`denoising/quantization`](https://github.com/curvedinf/k-splanifolds/tree/main/experiments/denoising/quantization) (historical INT8 internal arithmetic + from-scratch QAT) | Yes | FP32: "QAT gradients and master parameters remain floating point" |
| [`nanogpt_ks2`](https://github.com/curvedinf/k-splanifolds/tree/main/experiments/nanogpt_ks2) (GS32 W8A8 + FP16 nonlinearities) | Yes | FP32: "fused first-order FP32 surrogate backward" |

GS128 grouping exists only in the CPU paths of k-splanifolds; its GPU Triton fused
kernels are hardwired to GS32 (fused quantization epilogues require 32-feature tiles).
GPU GS128 INT8 does exist as a *serving* grid in int8-vllm (Section 8.1).

**Consequences for ksdraft:**
- The ks3_int8 fused-backward pattern is the structural template; its backward GEMMs
  must be moved to INT8 operands (INT32 accumulators), dequantizing to FP16 only for
  the returned gradient tensors that the optimizer consumes.
- Quantization points needed per backward stage: incoming-gradient requant for each
  backward GEMM; STE through weight/activation rounding as in ks3_int8.
- Rosa's counterfactual backward (currently FP/CPU) and KS3's geometry VJP must
  compose with the INT8 backward regime.

**Open items under this decision:**
- Kernel path: GPU (retile Triton fused kernels GS32 -> GS128) vs CPU (AVX-512 VNNI,
  where GS128 INT8 already exists but only as inference exports). Rosa backward is
  CPU-only today; KS3 fused kernels are GPU-only today.
- Master-weight / optimizer-state precision (FP16 masters with FP16 grads, or FP32
  masters) — undecided.

### 8.1 PTQR — INT8 checkpoint production methodology (borrowed from int8-vllm)

**Source:** the user's INT8-native vLLM fork [curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm)
(+ sibling [curvedinf/int8-aiter](https://github.com/curvedinf/int8-aiter)), built for
4x MI100 (gfx908, XGMI). Local authoritative checkout:
`/home/chase/Projects/vllm-upstream-bugs/fork-int8-vllm` (see its AGENTS.md and
`docs/recipes/README.md`; A/B ledger at `docs/recipes/surface_experiments_ledger.jsonl`).

**Method (from the [PTQR draft card](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR)):**
PTQR = post-training quantization with rounding-dither retraining. Fine-tunes model
weights directly under the deployed quantization:
- Student's forward pass replicates the **exact INT8 serving kernels** — the G128
  weight grid, dynamic per-token activation quantization, INT8 KV cache.
- Distillation against a **frozen BF16 teacher** (layer capture via
  `scripts/bf16_layer_capture.py`).
- **Geometrically annealed dither** on each rounding decision: early training
  explores neighboring grid points; late training is bit-identical to serving.
- Retrained weights re-quantized and exported through the same GPTQ GS128 wrapper
  (format- and config-identical to the one-shot original).

**Measured results (why it's worth borrowing):**
- Target (Qwen3.8-27B PTQR-R10S60): serving-gate KLD 0.0069 vs 0.0110 for deployed
  one-shot GPTQ; 42/52 greedy agreement vs 38/52; acceptance 3.88/13 ≥ BF16
  baseline 3.67.
- Draft (DFlash2 PTQR rung-1, 35 G128 quant payloads): fidelity gate top-1 mismatch
  0.042 over 120 probes — **identical to the BF16 drafter** (zero additional loss
  from quantization); acceptance 3.83/13 vs 3.88/13 BF16 at 42k context.
- INT8 KV (`int8_block_g128`) halved KV noise vs per-token/head INT8; fp16 draft KV
  measured broken (mean_k 1.01 — all drafts rejected).

**Toolchain to borrow (in the fork's `scripts/`):** `ptqr_train_target.py`,
`ptqr_train_draft.py`, `ptqr_export_gptq.py`, `ptqr_wrapper_export.py`,
`ptqr_draft_export.py`, `ptqr_stage_replay.py`, `ptqr_stage2_replay.py`,
`ptqr_round_replay.py`, `ptqr_control_base.py`, `ptqr_draft_cache.py`,
`bf16_layer_capture.py`, `qat_battery.py`.

**Also borrowable from the same fork (serving stack + doctrine):**
- INT8 serving primitives: AITER CK/Triton W8A8 GEMMs for GS128, fused
  round-to-nearest activation quant, INT8 KV cache G4–G128, INT8 embeddings/LM head,
  GS128 INT8 residual streams, custom XGMI all-reduce; DFlash/DFlash2 and LiLiCorr
  speculative decoding with INT8 draft projections and optional INT8 conv taps;
  adaptive speculative-token budgets (EAGLE-Harness-adjacent). A natural serving
  target for ksdraft, though Rosa+KS3 draft kernels would be new work in the fork.
- **Float-exception doctrine:** measured exceptions stay floating (GDN/Mamba
  recurrent state FP32 — int8 mamba failed quality; selector codebooks; draft
  ctx-KV projection on a bf16 > fp16 > int8 ladder), and every dtype change on a
  draft surface passes an acceptance gate (greedy bench, acceptance within ±1pt).
  Mirrors ks3_int8's "recurrent state stays FP32" contract; adopt the same
  gate-before-quantize rule for Rosa's discrete-symbol surfaces.

**Relation to Section 8:** Section 8 is *from-scratch* fully-INT8 training; PTQR is
the complementary *post-training / rounding-dither retraining* recipe for producing
INT8 checkpoints that match BF16 fidelity, including for draft models. ksdraft can
use both: native INT8 training from scratch, PTQR-style exact-serving-kernel student
+ annealed dither + fidelity gates for final checkpoint production.

---

## 9. Target model and draft sizing — DECIDED: Flint 40B/2.5A target, ~1B draft

**Decision (2026-11-08):** the target is **Recursive Flint**; the ksdraft draft is
sized at **~1B parameters** (depth-heavy allocation), with the EAGLE-Harness-style
cost model as the final arbiter of block length / fan-out / refinement budget.
Concrete shape decided the same day and corrected to the RWKV-8-only backbone
(see "Draft shape" below): 24 layers, d=1792, head_size 64, 597 one-bit Rosa
streams, 6,869 rank-1 KS3 charts, tied embeddings, 978,138,016 total (~0.98B).

### Flint — target model facts
- [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint): decoder-only MoE
  transformer for coding agents. **39.8B total, ~2.5B active per token** (~2.2B body
  + head); "40B-A2.5B" designation.
- 49 layers: 2 dense + 47 MoE; 256 routed experts, 6 selected per token + 1
  always-active shared expert.
- Attention: 35 sliding-window layers (window 513) + 14 full-attention layers; 32
  query / 4 KV heads, head dim 128; per-head QK RMSNorm + learned attention sinks;
  RoPE on sliding-window layers only.
- Context: 262,144 documented (the int8-vllm Flint recipe runs C12 with a 512k
  per-request cap, ~4.4M shared tokens). Vocab 150,144. Apache 2.0. BF16 ~80 GB.
- Evaluations: Terminal-Bench 2.1 54.9%, SWE-bench Pro v2 89.7%, SWE-bench Verified
  74.2%. Recommended sampling: temperature 1.0, top-p 0.95, top-k 20.
- Already first-class in int8-vllm: the Flint recipe ships a PTQR INT8 target + draft
  (built from flint + Flint-DFlash2) at C12 — target-side serving exists; ksdraft
  slots in as a new draft.

### Sizing rationale (~1B)
- **Against active, not total, parameters:** a verify round costs ~2.5B-active
  compute. A 1B dense draft is 1/2.5 of the target in FLOPs — 10x heavier than the
  classic 1/20–1/30 dense-ratio window. That is tolerable here for three reasons:
  1. **MoE verify is memory-bandwidth-bound at serving batch.** Each round the
     target streams 6-of-256 experts x 47 layers + shared experts + attention
     weights; the draft deterministically reads its ~1 GB (INT8). In bandwidth terms
     (the actual decode bottleneck on MI100/XGMI) the draft/target ratio is far below
     the FLOPs ratio.
  2. **JetSpec++ pipelining** hides draft wall-clock entirely while draft round ≤
     verify round; the bandwidth picture leaves slack for the refine + foresight
     passes.
  3. **The Rosa+KS3 draft is GEMM- and state-light**: hard retrieval (no attention
     GEMMs), constant state, fused INT8 kernels.
- **Direct evidence that ~400M is too small for this target:** the existing
  [Flint-DFlash2 draft](https://huggingface.co/incoai/Flint-DFlash2) (~400M BF16 /
  ~800 MB) reaches only τ ≈ 3.5–4.5 (block 8; GSM8K 4.23, MT-Bench 3.49) against
  this same target. DSpine-family results (τ ≈ 7.8 avg at T=0, block 16, on an 8B
  target) show the τ 6–10 regime is reachable with intra-block conditioning; the 1B
  budget plus DSpine injection + JetSpec self-refinement is the credible path there.
- **Sampling regime:** Flint serves at temperature 1.0 / top-p 0.95 — the
  high-entropy regime where acceptance drops (~1 τ at T=1 vs T=0 in DSpine's
  tables). Extra draft capacity defends τ exactly where it is most at risk.
- **Depth over width:** DSpine's causal chain unfolds over network depth; a deeper
  1B Rosa+KS3 stack gives the injection chain more room than a wide one.

### Constraints and gates
- **Upper bound:** past ~1.2–1.5B the draft round (draft sweep + self-refine +
  foresight adapter) risks exceeding the Flint verify round under load, where MoE
  expert reuse improves and verify becomes more FLOP-efficient. The harness must
  measure draft-round vs verify-round time at C1 and C12 and shrink block length /
  fan-out / refinement iterations, not just tree shape, under load (EAGLE-4's
  load-adaptive lesson generalized).
- **Lower bound:** below ~600M the Flint-DFlash2 τ≈4 datapoint suggests leaving
  acceptance on the table.
- **Memory:** 1B INT8 ≈ 1 GB weights + O(1) Rosa state per sequence (no drafter KV);
  negligible next to the target across 4x 32 GB MI100.

### Draft shape (DECIDED 2026-11-08; corrected same day to the RWKV-8-only backbone)

**Correction (2026-11-08):** the backbone is RWKV-8 proper - the Rosa QKV
mixer IS the time-mixing layer, with KS3 decoders. The RWKV-7 delta-rule
component was removed (the RWKV7+ROSA sum was the k-splanifolds experiment's
hybrid compromise, not this architecture). With the Rosa-only mixer the
per-layer cost drops, so the width moved 1536 -> 1792 to preserve the
decided ~1B envelope; depth stays 24.

**Tracking the RWKV World ~1B class.** The World lineage is consistent: every ~1.5B
model (RWKV-4 1B5, RWKV-5/6 Eagle/Finch 1B6, RWKV-7 G1.5B) is 24 layers x d=2048,
head_size 64 - verified directly on
[RWKV/v6-Finch-1B6-HF](https://huggingface.co/RWKV/v6-Finch-1B6-HF) config
(`hidden_size 2048, num_hidden_layers 24, head_size 64`); the
[RWKV-7 paper](https://openreview.net/pdf/74011cffc1082ac9394afde92802c810c51baabf.pdf)
lists the World ladder 0.1B/0.4B/1.5B/2.9B with 24 layers through 1.5B and 32 above.

| Model | Params | Layers | d_model | head_size |
| --- | --- | --- | --- | --- |
| RWKV World 0.4B | 0.4B | 24 | 1024 | 64 |
| RWKV-6 Finch 1B6 (verified) | 1.6B | 24 | 2048 | 64 |
| RWKV-7 G1.5B | 1.5B | 24 | 2048 | 64 |
| **ksdraft (decided)** | **~0.96B core** | **24** | **1792** | **64** |

**Decided configuration (d=1792, L=24):**

| Component | Setting | Params/layer |
| --- | --- | --- | --- |
| Rosa QKV mixer (the RWKV-8 time-mix) | 597 one-bit streams (d/3, matching the reference's 32 at d=96); 3 biased in-proj + 1 biased out-proj, 3 time-mix vectors, payload scales | 4,288,852 (4ds + 4d + 4s) |
| KS3 decoder | 6,869 rank-1 charts (d*368/96, matching both k-splanifolds precedents) | 24,673,448 |
| Per-layer core | | 28,962,300 |

Budget: 24 x 28,962,300 = 695,095,200 layers + 269,058,048 tied embedding
(Flint vocab 150,144 x 1792) = **964,153,248 core**; + DSpine injection and
LayerNorm extras 13,984,768 = **978,138,016 total (~0.98B)** - inside the
decided ~1B envelope and below the 1.2-1.5B harness ceiling.

**Why this shape:**
- **24 layers is the family constant** at this size class, and depth is what
  DSpine's injection chain needs (causal conditioning unfolds over depth).
- **d=1792 interpolates the family ladder** (0.4B: 1024, 1.5B: 2048); Flint's
  2.3x-larger vocab than RWKV's is paid in the tied embedding (269M).
- **Rosa streams scale with width** (s = d/3 -> 597 one-bit streams);
  bit-width per stream stays an ablation knob (1-bit x many streams is the
  validated configuration).
- **KS3 charts at d*368/96** matches both k-splanifolds precedents.
- **The k-splanifolds hybrid's 2048D decoder-only side residual is NOT carried
  over** (experiment scaffolding at 0.57M params; open ablation).
- Embedding and LM head tied (untying costs +269M - not recommended).

Open ablation knobs (not blocking): Rosa streams vs bits, KS3 chart density,
side-stream presence, per-layer allocation mix.

**Scale caveat (standing):** Rosa and KS3 were exercised at <=96D width in the
references; at 1792D the budget arithmetic is exact but training behavior is
what the MI100 runs will establish.

---

## 10. Training execution — DECIDED: live INT8 teacher on MI100; teacher-free development

**Decision (2026-11-08):** the teacher for the actual training runs is the **deployed
PTQR INT8 Flint target**, running live on the 4x MI100 server alongside draft
training — not a precomputed BF16 cache. Development happens teacher-free on the dev
box (AMD RX 7900 XTX, 24 GB). Serving pauses during training sessions.

### Memory math (why live INT8 teacher fits)
- PTQR INT8 Flint: 39.8B x 1 byte + GS128 scales ≈ ~40 GB → **~10 GB/GPU under TP4**
  (the BF16 case at ~20 GB/GPU did not fit; INT8 does).
- Draft training per GPU: int8 weights ~1 GB + masters 2.1–4.1 GB + Adam states
  2–8.2 GB + FP16 grads ~2 GB ≈ **13.5–15.5 GB/GPU** (knobs if tight: FP16 masters,
  8-bit Adam states, gradient checkpointing).
- Total ≈ 24–26 GB/GPU of 32 GB — fits with activation headroom. Transient teacher KV
  during training-time forwards: INT8 G128, ~1 KB/token/layer on the 14 full-attention
  layers (the 35 sliding-window layers are bounded by the 513 window); chunk
  sequences to bound it.
- **Recommended topology:** one training process owning both — load the PTQR INT8
  Flint weights via the int8-vllm loader and run teacher forwards inline/async —
  rather than co-scheduling a separate vLLM server plus a training process on the
  same GPUs (one memory budget; no static-fraction split to tune).

### Why live beats a static cache for this design
- **PTQR doctrine:** the draft learns against the exact deployed verifier behavior
  (INT8 rounding, INT8 G128 KV) — the acceptance-relevant distribution. Same
  exact-serving-kernel-student principle as PTQR itself.
- **JetSpec foresight requires it:** the foresight adapter is supervised on target
  states conditioned on verification outcomes of *the draft's own proposals* — those
  depend on the evolving draft and cannot come from a static precomputed cache.
  (DFlash-style dual-source conditioning on fixed corpus sequences is cacheable;
  draft-dependent supervision is not.)
- No cache format/storage problem; signal stays fresh as the draft evolves.
- Accepted tradeoff: compute contention — teacher forwards (2.5B-active, INT8,
  no backward) interleave with training steps; async/stream overlap mitigates.

### Development environment (dev box, teacher-free) — confirmed 2026-11-08
- Machine: RX 7900 XTX (gfx1100, 24 GB), ROCm 7.0.1, 24-core CPU, 61 GB RAM, 1.2 TB
  NVMe free. Environment: pyenv `rocm7` — torch 2.10.0+rocm7.0, triton 3.7.1, GPU
  access verified. MI100 stack pins torch 2.11.0+rocm7.1 / triton 3.6.0: acceptable
  skew; final kernel validation on MI100.
- gfx1100 (RDNA3) vs gfx908 (CDNA1) lower `tl.dot` int8 paths differently → the dev
  box is for **correctness only**; all performance gates run on MI100.
- 61 GB RAM cannot host Flint either (BF16 80 GB; INT8 ~40 GB + activations) — the
  dev box never runs the real teacher. It develops against (a) synthetic teacher
  fixtures behind the real interface, and (b) a surrogate teacher (Qwen3-8B BF16,
  ~16 GB, fits the card) to exercise conditioning→distill→foresight end-to-end at
  reduced fidelity.

### Teacher-free dev workstreams (build order)
1. Rosa Triton port (forward + counterfactual backward), parity-gated against the
   C++ exact engine running locally on the dev CPU.
2. KS3 GS128 retile (GS32→GS128 fused epilogues) + INT8 backward GEMMs with FP16
   gradient handoff; parity vs `ks3.py` FP32 reference + analytic VJP. Full-width
   single-cell tests (18M params) are trivial on 24 GB.
3. DSpine injection + transfer-space supervision (targets are token embeddings).
4. Full 1.03B assembly; ~13.5–15.5 GB optimizer/weight states + checkpointed
   activations → smoke-trains on 24 GB at micro-batch.
5. CE pretraining stage — genuinely teacher-free; this is phase-1 training, not just
   a test.
6. Teacher client interface with fixture mode (the dev-box stand-in for the live
   MI100 teacher): hidden-state queries at the injection layers, top-k logits,
   verification rollouts; surrogate-teacher backing.
7. First MI100 session (off-hours): load PTQR INT8 Flint in-process, validate the
   live-teacher path end-to-end, then scale training.

---

## 11. Training regime — DESIGNED 2026-11-08

**Philosophy (from the drafting-model survey):** two-phase teacher-forced
distillation — mine data and hidden states from the target, then train the drafter
to mimic it — never raw-text from-scratch training as the main event. ksdraft
adapts the shared recipe in four ways: (a) the Rosa sweep is *causal* (DSpine's
predecessor-conditioned paradigm, not DFlash's masked bidirectional prediction),
(b) the teacher is the **live deployed PTQR INT8 Flint** (Section 10), so the
student always trains against the exact verifier distribution and draft-dependent
supervision is available on demand, (c) training is **native INT8 end-to-end**
(Section 8) with PTQR rounding-dither annealing woven through the stages, and
(d) JetSpec foresight and self-refinement are trained as explicit late stages.

### Stage 0 — Architecture warm-up (teacher-free; dev box or MI100)
- Short CE warm-start of the Rosa+KS3 backbone on raw code+text — **≤5% of token
  budget**, checkpoint discarded before distillation. (Deliberate deviation from the
  field's no-scratch recipe: the architecture is novel at 1536D and the INT8
  training system itself needs a stable shakedown; the k-splanifolds evidence base
  is all ≤96D.)
- Initialize the draft's tied embedding from an SVD projection of Flint's embedding
  table (vocab is shared; width differs) — token semantics for free, then trained.

### Stage 1 — Mining (live teacher; MI100)
- Prompts: coding-agent corpora in the DCOCO style (Nemotron-PT-v2-class post-
  training data, CodeAlpaca-class, agentic/tool-use traces) plus long-context
  repository text — deliberately heavy on code repetition, where Rosa's
  longest-suffix retrieval is the strongest inductive match.
- **Tier-5 512k needle-in-a-haystack documents** (added to the fork's PTQR corpus
  builder `scripts/ptqr_flint_corpus.py`, 2026-11-09): 512k-token C4 haystacks
  with a planted fact ("access code for PROJECT-XXXXXX is ZKYYYYYYYY"), chat-
  formatted as read-and-retrieve tasks; needles half within the last 16k tokens
  (feasible-window retrieval), half at uniformly random depth (full-document
  exposure). For the **draft**, scale `--n-xlong` well beyond the target's default
  of 8 (e.g., 64–256 docs ≈ 32–128M tokens — still a small slice of the 5–10B
  budget) and mine the teacher's continuations on them: the needle-answer legs are
  a pure copy-through-retrieval task — Rosa's exact mechanism — and the draft
  should see the same corpus tiers the target was PTQR-trained on (distribution
  match between teacher training and student distillation).
- On xlong docs, anchor sampling spans the full depth: stratify anchors near
  planted facts, near the closing query, and at random depth, so dual-source
  conditioning carries the retrieved fact into the block that must emit it.
- Continuations **regenerated by the PTQR INT8 Flint** in two modes: greedy
  thinking-off (the DCOCO recipe) and sampled at Flint's serving parameters
  (T=1.0, top-p 0.95, top-k 20) — the acceptance regime is T=1, so mine both.
- Mining caution from the fork's ledger (open hypothesis, 2026-11): the teacher may
  never have been trained on BOS-prefixed raw text — feed the teacher chat-template-
  formatted legs (as the corpus builder does) rather than raw concatenated slices.
- Capture at each anchor: multi-layer teacher hidden states — tap ~5 layers spread
  across Flint's 49 (e.g., {6, 14, 22, 30, 38} + final), the EAGLE-3/DFlash
  multi-layer fusion analog — plus top-k logits (k=32: IDs + FP16 probs).
- Mining builds the *initial* corpus only; Stages 4–6 also query the live teacher
  on draft rollouts, which no static cache can contain.
- **Standing eval gate:** acceptance-by-context-depth (long-context decay) tracked
  from Stage 2 onward, with the NIAH legs as a held-out probe — the fork's
  draft-quality doctrine already monitors decay; Rosa's constant state is the
  mechanism meant to beat it across the 512k per-request window the Flint recipe
  serves.

### Stage 2 — Drafter distillation (core stage)
- **Objective:** teacher-forced block CE over causal-sweep blocks (anchor + block of
  16). Rosa's recurrence supplies intra-block causality natively; DSpine gated
  adjacent injection adds the learned depth-conditioned chain. No bidirectional
  mask exists to train.
- **Anchor sampling:** random anchor subset per sequence (DFlash practice). With
  constant-state recurrence, anchors are cheap — sample densely (≥1 anchor per 64
  tokens).
- **Transfer-space auxiliary loss at every layer** (DSpine): each layer's predicted
  feature aligned to the next token's embedding — this is what makes depth
  injection and predecessor-conditioned decoding coherent at inference.
- **Dual-source conditioning:** fused teacher states feed the draft; the natural
  Rosa analog of H-Spec's seeding is projecting the last-token target state into
  the recurrent state at block start.
- Small-weight KL to teacher top-k logits as a secondary loss.
- **Optimization:** AdamW, cosine schedule, peak ~3e-4 (DCOCO used 6e-4 at ~400M;
  scale down modestly for 1B), warmup, grad-norm clip 1.0. Budget: order 5–10B
  teacher tokens.
- **Numerics:** fully INT8 forward/backward, FP16 gradient handoff; rounding dither
  ON (PTQR exploration phase — early training explores neighboring grid points).

### Stage 3 — Corrector stage (conditional; DCOCO fallback path)
- Only if the DCOCO-style parallel contextual corrector is activated (fallback if
  DSpine injection under-delivers): joint drafter + corrector training after the
  drafter warm-up, total loss = drafter loss + corrector transition loss
  (DCOCO: 6 warm-up epochs + 6 joint; frozen embeddings during joint training).
- Primary plan skips this stage — DSpine injection is in the backbone from Stage 2;
  candidate top-k retention still feeds the scheduler either way.

### Stage 4 — Foresight adapter (JetSpec; EAGLE-4-style staging)
- **Freeze the drafter** (EAGLE-4 pattern: stage-1 parameters frozen while a
  lightweight head trains); the foresight adapter is low-rank/parameter-light.
- Train it to predict target hidden states + anchor tokens under anticipated
  verification outcomes, supervised by the live teacher's *actual* states on the
  draft's own rollouts. Loss: CE on anchor token + cosine/MSE on predicted states.
- This stage is the concrete payoff of the live-teacher decision: foresight
  supervision is draft-dependent and cannot be pre-mined.

### Stage 5 — Self-refinement (JetSpec++ intra-block)
- Unfreeze the drafter (or LoRA it); train the refine iterations on the model's
  *own* drafted blocks (draft-forced), verified by the live teacher — closing the
  exposure-bias gap that pure teacher forcing leaves.

### Stage 6 — Acceptance-aware fine-tuning
- With the live teacher, acceptance is **measured, not simulated**: Window Total
  Variation loss for the sampling regime (Flint serves at T=1.0/top-p 0.95, so WTV
  is the operative objective; EAL covers greedy), optionally a short GRPO stage
  with real accepted-length reward. Short, low LR, dither annealing toward zero
  through this stage.

### Stage 7 — PTQR consolidation and export gates
- Anneal rounding dither fully to zero → training becomes bit-identical to the
  serving kernels; brief consolidation at dither = 0.
- **Gates (PTQR battery):** quantized drafter's top-1 mismatch identical to the
  unquantized drafter (zero additional loss from quantization); acceptance within
  ±1pt of the unquantized drafter on the TP4/C12 greedy bench; then export through
  the serving wrapper into int8-vllm.
- **Harness calibration:** measure draft-round vs verify-round at C1 and C12; set
  block length, fan-out, and refinement budget from measured costs (Section 9's
  gates).

### Evaluation targets
- Beat the Flint-DFlash2 baseline (τ ≈ 3.5–4.5 at block 8): τ ≥ 5 at block 8.
- Ambition: τ 6–10 at block 16, T=0 — expect roughly −1 at T=1 per DSpine's tables.
- End-to-end: ≥1.5x AR throughput at C1 and C8 on the MI100 INT8 stack, lossless.

### Hardware mapping
- Stages 0–2 smoke-test on the dev box (fixture/surrogate teacher); all distillation
  stages run on MI100 co-located with the live INT8 teacher per Section 10's memory
  budget (~24–26 GB/GPU of 32 GB).

---

## 12. Product & distribution goals — DECIDED 2026-11-08

Two deliverables beyond the research model itself:

**(A) A draft-creation toolkit** — creating drafts for *new target models* must be
usable by people who did not build this project.
**(B) An optimized torch draft package** — a portable model implementation others can
drop into serving stacks (vLLM, SGLang, ...).

### A. The toolkit (`ksdraft` CLI, config-driven)

The Section 11 pipeline is target-agnostic machinery with Flint-specific parameters;
the toolkit packages the machinery and makes the parameters config.

| Command | What it packages |
| --- | --- |
| `ksdraft size` | Draft-shape proposal from target specs (active/total params, vocab, context) — codifies the Section 9 methodology: active-param ratio gates, RWKV-family tracking, embedding-budget arithmetic |
| `ksdraft mine` | Corpus tiers (incl. tier-5 NIAH) + teacher mining through the teacher-client interface |
| `ksdraft train` | Stages 0–7 with fixture/surrogate modes for teacher-free dev |
| `ksdraft gate` | PTQR battery + acceptance gates + harness calibration (C1/C12 draft-vs-verify measurement) |
| `ksdraft export` | Serving formats: HF-style repo (safetensors + config + modeling code) and engine adapter payloads |

- **The target adapter is the seam:** tokenizer/vocab, hidden-state tap indices,
  teacher client, serving export format. Flint is the first adapter; core code never
  branches on target identity (same discipline as the voodoo-dyn-quant arch-adapter
  registry and int8-vllm's model adapters).
- **Teacher clients:** in-process PTQR INT8 teacher (int8-vllm loader) where the fork
  supports the target; HF fallback (frozen BF16, arbitrary target) accepting the
  quantization mismatch; fixture mode for dev. The live-vs-cache question is a client
  property, not a pipeline property.
- **Precedents:** eagle-harness (PyPI tool + per-engine plugins, calibration CLI),
  AngelSpec (documented drafter family over one training pipeline).

### B. The torch package (droppable draft model)

- **`KSDraftModel` (torch.nn.Module)** with a small proposal API:
  `propose(target_states, tokens, block_len, topk) -> candidates, logits, refined`
  — DSpine injection, candidate retention, and refinement iterations are internal;
  the foresight adapter is an optional component. Lossless verification stays in the
  engine; the draft only proposes.
- **Configurable dtype, three tiers, one checkpoint (decided 2026-11-08):**
  - `int8` — the intended, assuredly-supported target. Full fusion and the deepest
    optimization: the ks3_int8 pattern (W8A8 GS128, fused input-GEMM+geometry,
    fused quantization epilogues, INT8 tanh-LUT nonlinearities, fused backward for
    training) plus int8-vllm's serving fusions (fused round-to-nearest activation
    quant, fused all-reduce/RMSNorm/quant epilogues, GS128 INT8 residual streams).
    Float exceptions per the doctrine: Rosa/recurrent state, warp scalars and
    accumulators stay FP32 inside the int8 path.
  - `bf16` — the generic fallback: same model graph, plain torch GEMMs (optionally
    `torch.compile`), runs on any device including CPU and inside engines without
    int8 kernels; doubles as the parity reference the int8 tier is gated against.
  - `fp32` — parity/debug reference (the `ks3.py` pattern) used by the test battery.
  - One checkpoint of master weights; GS128 packs and scales are derived at
    load/export, so a draft trains once and serves in any tier. Training itself
    (Section 8) is INT8-native regardless — the dtype axis governs the served
    forward package.
  - **Testbed alignment:** the dev box's RX 7900 XTX (gfx1100) has usable INT8 dot
    paths, so the entire fused int8 tier is exercised *functionally* on the dev
    machine — correctness gates run there — while MI100 (CDNA1: 2x INT8 matrix
    rate at half bandwidth, the native dtype of that stack) validates performance.
    The gfx1100-vs-gfx908 lowering difference makes the dev box a genuine
    cross-check of kernel portability, not a formality.
- **Distribution: the HF-repo pattern** proven by Flint-DFlash2 and DSpine — publish
  checkpoint + custom modeling code + serving kit as a repo, so engines consume it
  via `--speculative-draft-model-path` style flags.
- **Engine adapters, in order:** (1) int8-vllm first-class — the fork's
  DFlash/DFlash2 speculative method shows the exact integration shape (draft
  projections, candidate handling, INT8 KV for the draft); (2) SGLang and vLLM
  adapters following the same pattern the ecosystem already accepts (SGLang's DFLASH
  PR #45624; eagle-harness's plugin set).
- **Honest limit:** the in-process INT8 teacher requires int8-vllm support for a
  given target; new targets without it use the HF BF16 fallback teacher (or a
  one-shot INT8 teacher), trading a little acceptance fidelity for universality.

### Consequences for earlier sections
- Section 10's teacher-client interface is now the toolkit's adapter seam — design
  it as a public interface from day one.
- Section 11 Stage 7 gains export targets beyond int8-vllm: HF-repo format + engine
  adapter payloads.
- Section 10's dev workstreams gain a standing constraint: build the repo as an
  installable package with a stable API from the start, not as research scripts that
  get repackaged later.


### 8.2 Lessons from the fork's three INT8 draft implementations

The int8-vllm fork ships three INT8 draft paths; what worked and what stayed
floating (source: the fork's models and commit series on origin/main):

| Draft | INT8 adoption | Measured exceptions / notes |
| --- | --- | --- |
| DFlash2 (Qwen3.8-27B, PTQR rung-1) | 35 G128 quant payloads + 12 dequant-grid bf16 surfaces on the deployed wrapper; fidelity gate top-1 == BF16 | selector codebooks and the draft ctx-KV projection stayed bf16 (ladder bf16 > fp16 > int8); GDN state fp32 |
| LiLiCorr (Nemotron Bolt) | GS128 INT8 residual streams, activation epilogues, draft convolutions, INT8 draft projections + optional INT8 conv taps | GS128 residual transport end-to-end |
| Flint draft (Flint-DFlash2) | near-universal: int8 selector codebooks via gather-dequant with per-row scales; int8-only dense/shared/draft MLP gate_up with fused SwiGLU+quant; int8 residual stream end to end, no bf16 residual path | PTQR trains the router surface and recovers the cold KLD — PTQR-retraining beats one-shot quant on every surface that initially hurt |

Lessens adopted by ksdraft: (1) small tables (codebooks) quantize fine with
per-row-scale gather-dequant — the earlier bf16 exception fell in the Flint
generation; (2) fuse quantization into epilogues (SwiGLU+quant, norm+quant)
rather than materializing intermediates; (3) when a surface degrades under
one-shot int8, PTQR-retrain it before declaring it a float exception; (4) the
float-exception list shrinks with each generation — int8 can be widely
adopted (user directive), with acceptance gates as the arbiter.

---

## 13. Implementation status (2026-11-08)

The training environment is built and MI100-ready (see MI100.md). Decision ->
implementation map:

| Decision (section) | Implementation | Validation |
| --- | --- | --- |
| Draft shape (9) | `ksdraft/config.py` — exact param formulas locked | `tests/test_config.py` (exact integers) |
| Rosa mixer + match (5) | `ksdraft/model/rosa.py` — torch reference | brute-force oracle on random bit sequences |
| Rosa counterfactual backward (5, 7) | `ksdraft/kernels/rosa_credit.py` — exact-credit engine integration | engine forward == torch match at atol=0; symbol-projection gradients verified |
| KS3 Horner spine (5) | `ksdraft/model/ks3.py` — canonical port | exact parity vs k-splanifolds `ks3.py` |
| All-INT8 KS3 (8, 12) | `ksdraft/kernels/ks3_int8.py` + fused `ks3_triton.py` | parity vs `ks3_int8.py` on its GS32 grid; STE grads cos 1.0000 on geometry |
| GS128 W8A8 (8) | `ksdraft/kernels/quant.py`, `gemm.py` | triton-vs-torch parity on gfx1100; partial-group padding |
| DSpine injection (2, 11) | `ksdraft/model/spine.py` | shape/shift/zero-init tests |
| Assembly + dtype tiers (12) | `ksdraft/model/draft.py` | identity-block init; bf16 forward; teacher seam; meta-device budget |
| Corpus incl. tier-5 NIAH (11) | `ksdraft/data/` | strata + format locked vs the fork's generator |
| Teacher clients (10, 12) | `ksdraft/teacher/` — Fixture + HF backends | determinism + simplex checks; HF behind lazy import |
| Staged trainer (11) | `ksdraft/train/` — warmup/distill, transfer-space + top-k CE | loss-decrease smokes (CPU + gfx1100 via CLI) |
| Gates + export (11 Stage 7, 12) | `ksdraft/gate.py`, `ksdraft/export.py` + CLI | battery passes; safetensors roundtrip incl. tying |
| MI100 run guide (10) | `MI100.md` | — |

Follow-ups completed 2026-11-09: model-wide Int8Linear swap-in
(`model/int8_convert.py`, gate-checked), GPU-native exact Rosa credits
(`kernels/rosa_credit_gpu.py` — cosine 1.0000 vs the engine's first-order
gradients, forward bitwise-exact, T <= 63 windows), and trainer stages 4-7
(`train/stages.py`: foresight adapter, draft-forced refinement, WTV
acceptance, PTQR dither annealing + fidelity gates).

Remaining boundaries: the GPU credit path is exact-but-windowed (T <= 63;
beyond that use the CPU engine or the future suffix-automaton kernel);
KS3 `ddown` backward cosine 0.82 vs the fp32 reference (float-exception
candidate); int8 embedding/head (per-row-scale gather) still staged.
