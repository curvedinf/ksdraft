"""Draft model configuration.

Encodes the decided ksdraft shape (RESEARCH.md Section 9, "Draft shape
DECIDED 2026-11-08") and the dtype tiers (Section 12). Standard-library only
so the config and its parameter-budget math are usable everywhere.

Parameter formulas per layer (Section 9 table):
  - RWKV-7-style delta-rule mixer: 5*d^2 (r,k,v,g,out) + 4*32*d (w/a LoRA
    pairs) + 6*d (time-mix vectors) ~= 5.03 d^2
  - Rosa QKV mixer: 4*d*s (three biased input projections d->s and one output
    projection s->d) + 3*d (time-mix vectors) + s (payload scales)
  - KS3 decoder: charts * (2*d + 6) + 2*charts (rank-1 subspaces; six spine
    coefficients per output coordinate, warp scale/shift per chart)
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_DTYPES = ("int8", "bf16", "fp32")

Dtype = str  # one of VALID_DTYPES; a plain alias keeps configs YAML/JSON friendly

# KS3 chart density: 368 charts at width 96 in the k-splanifolds reference
# (exactly 5888 charts at d=1536).
KS3_CHARTS_PER_WIDTH = 368 / 96

# Rosa stream density: 32 one-bit streams at width 96 in the reference
# (512 streams at d=1536).
ROSA_STREAMS_PER_WIDTH = 32 / 96


@dataclass
class KSDraftConfig:
    """Decided default: the ~1.03B draft targeting Flint (Section 9)."""

    vocab_size: int = 150_144
    layers: int = 24
    d_model: int = 1536
    head_size: int = 64
    rosa_streams: int = 512
    rosa_bits: int = 1
    ks3_charts: int = 0  # 0 -> round(d_model * 368 / 96)
    block_len: int = 16
    anchor_stride: int = 64
    dtype: Dtype = "int8"
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"dtype must be one of {VALID_DTYPES}, got {self.dtype!r}")
        if self.d_model % self.head_size != 0:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by head_size {self.head_size}"
            )
        if not 1 <= self.rosa_bits <= 64:
            raise ValueError("rosa_bits must be in [1, 64] (Rosa packed symbol width)")

    @property
    def n_heads(self) -> int:
        return self.d_model // self.head_size

    @property
    def charts(self) -> int:
        if self.ks3_charts:
            return self.ks3_charts
        return round(self.d_model * KS3_CHARTS_PER_WIDTH)

    # ---- per-layer parameter counts (exact integers) ----

    def mixer_params(self) -> int:
        # Faithful TimeMix port (k-splanifolds rwkv7_ks2 run.py): read/key/
        # value/output linears 4*d^2 + controllers (decay 64d, rate 64d,
        # gate 128d, value 64d — value_control on every layer here) + six
        # time-mix vectors 6d + five bias/scale vectors 5d + bonus d +
        # GroupNorm affine 2d.
        d = self.d_model
        return 4 * d * d + 334 * d

    def rosa_params(self) -> int:
        # Three biased input projections d->s (+3s), biased output s->d (+d),
        # three d-dim time-mix vectors (+3d), payload scales (+s).
        d, s = self.d_model, self.rosa_streams
        return 4 * d * s + 4 * d + 4 * s

    def ks3_params(self) -> int:
        d, c = self.d_model, self.charts
        return c * (2 * d + 6) + 2 * c

    def layer_params(self) -> int:
        return self.mixer_params() + self.rosa_params() + self.ks3_params()

    def embedding_params(self) -> int:
        p = self.vocab_size * self.d_model
        if not self.tie_word_embeddings:
            p *= 2
        return p

    def total_params(self) -> int:
        return self.layers * self.layer_params() + self.embedding_params()

    def summary(self) -> dict:
        return {
            "layers": self.layers,
            "d_model": self.d_model,
            "head_size": self.head_size,
            "n_heads": self.n_heads,
            "rosa_streams": self.rosa_streams,
            "rosa_bits": self.rosa_bits,
            "ks3_charts": self.charts,
            "dtype": self.dtype,
            "mixer_params_per_layer": self.mixer_params(),
            "rosa_params_per_layer": self.rosa_params(),
            "ks3_params_per_layer": self.ks3_params(),
            "layer_params": self.layer_params(),
            "embedding_params": self.embedding_params(),
            "total_params": self.total_params(),
        }
