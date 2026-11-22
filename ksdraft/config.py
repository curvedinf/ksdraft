"""Draft model configuration.

Encodes the decided ksdraft shape (RESEARCH.md Section 9, corrected
2026-11-08 to the RWKV-8-only backbone) and the dtype tiers (Section 12).
Standard-library only so the config and its parameter-budget math are usable
everywhere.

The backbone is RWKV-8 proper: the Rosa QKV mixer IS the time-mixing layer
(no RWKV-7 delta-rule component). Per-layer core parameters:

  - Rosa QKV mixer: 4*d*s (three biased input projections d->s and one
    biased output projection s->d) + 4*d (three time-mix vectors + output
    bias) + 5*s (three input biases + payload scales + learned empty payload)
  - KS3 decoder: charts * (2*d + 6) + 2*charts (rank-1 subspaces; six spine
    coefficients per output coordinate, warp scale/shift per chart)

DSpine adjacent injection and LayerNorms are tracked as model-level extras
(~1.5% of the total), not part of the core formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_DTYPES = ("int8", "bf16", "fp32")

Dtype = str  # one of VALID_DTYPES; a plain alias keeps configs YAML/JSON friendly

# KS3 chart density: 368 charts at width 96 in the k-splanifolds reference.
KS3_CHARTS_PER_WIDTH = 368 / 96

# Rosa stream density: 32 one-bit streams at width 96 in the reference.
ROSA_STREAMS_PER_WIDTH = 32 / 96


@dataclass
class KSDraftConfig:
    """Decided default: the ~1B RWKV-8 draft targeting Flint (Section 9)."""

    vocab_size: int = 150_144
    layers: int = 24
    d_model: int = 1792
    head_size: int = 64
    rosa_streams: int = 597
    rosa_bits: int = 1
    ks3_charts: int = 0  # 0 -> round(d_model * 368 / 96)
    block_len: int = 16
    anchor_stride: int = 64
    dtype: Dtype = "int8"
    tie_word_embeddings: bool = True
    # DFly2-style target-KV reuse: kv_dim > 0 replaces the Rosa mixer with
    # cross-attention over target-provided KV (no drafter-side KV cache).
    # kv_dim is the target's per-layer cached K/V width (Inkling-2-Mini:
    # 8 KV heads x head_dim 128 = 1024).
    kv_dim: int = 0
    n_kv_heads: int = 8
    kv_window: int = 1024

    def __post_init__(self) -> None:
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"dtype must be one of {VALID_DTYPES}, got {self.dtype!r}")
        if not 1 <= self.rosa_bits <= 64:
            raise ValueError("rosa_bits must be in [1, 64] (Rosa packed symbol width)")
        if self.kv_dim < 0:
            raise ValueError("kv_dim must be >= 0 (0 = Rosa backbone)")
        if self.n_kv_heads < 1:
            raise ValueError("n_kv_heads must be >= 1")
        if self.kv_window < 1:
            raise ValueError("kv_window must be >= 1")

    @property
    def n_heads(self) -> int:
        # head-structured surfaces (e.g., grouped norms) use head_size granularity
        return self.d_model // self.head_size

    @property
    def charts(self) -> int:
        if self.ks3_charts:
            return self.ks3_charts
        return round(self.d_model * KS3_CHARTS_PER_WIDTH)

    # ---- reuse-attention head geometry (Section 9 Inkling 0.5B shape) ----

    @property
    def n_kv_eff(self) -> int:
        # adapted KV heads, capped by the width's natural query-head count
        return max(1, min(self.n_kv_heads, self.d_model // self.head_size))

    @property
    def q_heads(self) -> int:
        # query heads rounded down to a whole number of KV groups (GQA)
        base = max(1, self.d_model // self.head_size)
        kv = self.n_kv_eff
        return max(kv, (base // kv) * kv)

    # ---- per-layer core parameter counts (exact integers) ----

    def rosa_params(self) -> int:
        # Three biased input projections d->s (+3s), biased output s->d (+d),
        # three d-dim time-mix vectors (+3d), payload scales (+s) and the
        # learned empty payload (+s).
        d, s = self.d_model, self.rosa_streams
        return 4 * d * s + 4 * d + 5 * s

    def reuse_attn_params(self) -> int:
        # Q and output projections (d <-> q_heads*head_size), plus the K and V
        # adapters from the target's cached KV width into the draft's KV heads
        # (kv_dim -> n_kv_eff*head_size). No biases, no drafter-side KV cache.
        q = self.q_heads * self.head_size
        ko = self.n_kv_eff * self.head_size
        return 2 * self.d_model * q + 2 * self.kv_dim * ko

    def mixer_params(self) -> int:
        return self.reuse_attn_params() if self.kv_dim else self.rosa_params()

    def ks3_params(self) -> int:
        d, c = self.d_model, self.charts
        return c * (2 * d + 6) + 2 * c

    def layer_params(self) -> int:
        return self.mixer_params() + self.ks3_params()

    def embedding_params(self) -> int:
        p = self.vocab_size * self.d_model
        if not self.tie_word_embeddings:
            p *= 2
        return p

    def total_params(self) -> int:
        return self.layers * self.layer_params() + self.embedding_params()

    def summary(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "layers": self.layers,
            "d_model": self.d_model,
            "head_size": self.head_size,
            "n_heads": self.n_heads,
            "rosa_streams": self.rosa_streams,
            "rosa_bits": self.rosa_bits,
            "ks3_charts": self.charts,
            "block_len": self.block_len,
            "dtype": self.dtype,
            "tie_word_embeddings": self.tie_word_embeddings,
            "kv_dim": self.kv_dim,
            "n_kv_heads": self.n_kv_heads,
            "kv_window": self.kv_window,
            "q_heads": self.q_heads,
            "mixer_params_per_layer": self.mixer_params(),
            "rosa_params_per_layer": self.rosa_params(),
            "attn_params_per_layer": self.reuse_attn_params(),
            "ks3_params_per_layer": self.ks3_params(),
            "layer_params": self.layer_params(),
            "embedding_params": self.embedding_params(),
            "total_params": self.total_params(),
        }
