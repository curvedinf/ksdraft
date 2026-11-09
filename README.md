# KSDraft

Parallel block-drafting draft models for speculative decoding.

Architecture:

- **DSpine depth-carried causal injection**
- **JetSpec++ foresight pipelining**
- **EAGLE-Harness-style cost scheduler**
- **RWKV-8 Rosa attention**
- **KS3 decoders**
- **Fully native INT8 forward and backward**
- **PTQR rounding-dither annealing**

Status: implementation in progress. The full architecture research, decisions, and
training-regime design live in [RESEARCH.md](RESEARCH.md) — it is the source of
truth for every design choice in this repo.

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
# Draft-shape budget for the decided configuration (RESEARCH.md Section 9)
ksdraft size
ksdraft size --layers 24 --d-model 1536 --vocab 150144 --rosa-streams 512
```

## Layout

```
ksdraft/
  config.py     # KSDraftConfig — the decided draft shape and dtype tiers
  cli.py        # ksdraft size | mine | train | gate | export
  model/        # Rosa mixer, KS3 decoder, DSpine injection, KSDraftModel
  kernels/      # INT8 tier: W8A8 GS128 fused kernels (int8 forward + backward)
  teacher/      # teacher clients: fixture | HF bf16 | in-process INT8 (int8-vllm)
  train/        # staged training regime (RESEARCH.md Section 11)
  data/         # corpus builders incl. tier-5 512k needle-in-a-haystack
configs/        # target adapters (flint.yaml first)
tests/
```

## Dtype tiers

- `int8` — intended target: W8A8 GS128, maximal fusion (training AND serving)
- `bf16` — generic fallback: same graph, portable kernels, parity reference
- `fp32` — debug/parity tier for the test battery

Training is INT8-native regardless of the serving tier.

## License

MIT — see [LICENSE](LICENSE).
