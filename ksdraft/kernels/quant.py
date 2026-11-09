"""GS128 symmetric INT8 quantization.

Contract (ks3_int8 lineage, retiled 32 -> 128 per RESEARCH.md Section 8):

- Symmetric per-group scales, group size 128 along the quantized dimension.
- Scales are stored FP16, rounded BEFORE encoding (encode and decode use the
  same value), bounded to [2**-24, 65504].
- Codes saturate to [-127, 127]; round-to-nearest-even (torch.round).
- Dimensions not divisible by 128 are zero-padded to the next group; padded
  lanes decode to 0.

Weights: per-output-row groups (quantize along the input dim).
Activations: per-token dynamic groups (also along the input dim).
"""

from __future__ import annotations

import torch

GROUP_SIZE = 128
SCALE_DTYPE = torch.float16
SCALE_MIN = 2.0**-24
SCALE_MAX = 65504.0


def _padded_length(n: int, group: int = GROUP_SIZE) -> int:
    return (n + group - 1) // group * group


def quantize_last_dim(
    x: torch.Tensor, group: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """x (..., K) fp -> (codes int8 (..., Kp), scales fp16 (..., Kp/group)).

    The padded tail (Kp - K lanes) quantizes to zero codes. `group` defaults
    to 128 (ksdraft decision); 32 matches the k-splanifolds reference grid
    for parity tests.
    """
    if not torch.is_floating_point(x):
        raise TypeError("quantize expects floating-point input")
    K = x.shape[-1]
    Kp = _padded_length(K, group)
    if Kp != K:
        x = torch.nn.functional.pad(x, (0, Kp - K))

    grouped = x.reshape(*x.shape[:-1], Kp // group, group)
    amax = grouped.abs().amax(dim=-1)
    scale = (amax.float() / 127.0).clamp(SCALE_MIN, SCALE_MAX)
    scale = scale.to(SCALE_DTYPE).to(torch.float32)  # round via FP16, then use fp32
    codes = torch.clamp(torch.round(grouped / scale[..., None]), -127.0, 127.0)
    return codes.to(torch.int8).reshape(*x.shape[:-1], Kp)[..., :K].contiguous(), \
        scale.to(SCALE_DTYPE)


def dequantize_last_dim(
    codes: torch.Tensor, scales: torch.Tensor, group: int | None = None
) -> torch.Tensor:
    """Inverse of quantize_last_dim (exact on the codes it produced)."""
    K = codes.shape[-1]
    if group is None:
        group = scales.shape[-1] and _padded_length(K) // scales.shape[-1] or GROUP_SIZE
        group = _padded_length(K) // max(scales.shape[-1], 1)
    Kp = _padded_length(K, group)
    if Kp != K:
        codes = torch.nn.functional.pad(codes, (0, Kp - K))
    grouped = codes.reshape(*codes.shape[:-1], Kp // group, group)
    return (grouped.float() * scales.float()[..., None]).reshape(*codes.shape[:-1], Kp)[..., :K]
