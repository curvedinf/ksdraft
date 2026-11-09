"""Int8Linear — W8A8 GS128 drop-in for nn.Linear (inference/QAT forward).

Weights are quantized once at construction (per-output-row groups along the
input dim); activations are quantized dynamically per token. The Triton path
runs on CUDA/HIP; every other device uses the pure-torch fallback with
identical numerics (same codes, fp32 dequant matmul).

This milestone covers the forward. The INT8 backward (INT8-operand GEMMs,
FP16 gradient handoff — RESEARCH.md Section 8) arrives with the training
integration; until then use `.fake_quant_forward` for STE-friendly fp32
autograd.
"""

from __future__ import annotations

import torch
from torch import nn

from .gemm import w8a8_gemm
from .quant import quantize_last_dim


class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # zeros (not empty): __init__ quantizes immediately, and uninitialized
        # memory can hold NaN bit patterns that would poison the codes.
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("weight_codes", None, persistent=False)
        self.register_buffer("weight_scales", None, persistent=False)
        self._quantize_weight()

    @classmethod
    def from_float(cls, module: nn.Linear) -> "Int8Linear":
        lin = cls(module.in_features, module.out_features, bias=module.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(module.weight)
            if module.bias is not None:
                lin.bias.copy_(module.bias)
        lin._quantize_weight()  # codes must reflect the copied masters
        return lin.to(module.weight.device)

    @torch.no_grad()
    def _quantize_weight(self) -> None:
        codes, scales = quantize_last_dim(self.weight.detach().float())
        self.weight_codes = codes
        self.weight_scales = scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._quantize_weight()  # cheap; keeps live fp32 masters authoritative
        x2 = x.reshape(-1, x.shape[-1])
        a_codes, a_scales = quantize_last_dim(x2.float())
        y = w8a8_gemm(a_codes, a_scales, self.weight_codes, self.weight_scales)
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

    def fake_quant_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STE path: fp32 GEMM on quantize-dequantized operands (autograd-safe)."""
        from .quant import dequantize_last_dim

        self._quantize_weight()  # live fp32 masters stay authoritative
        w = dequantize_last_dim(self.weight_codes.detach().float(),
                                self.weight_scales.detach()) \
            + (self.weight - self.weight.detach())
        x2 = x.reshape(-1, x.shape[-1]).float()
        codes, scales = quantize_last_dim(x2)
        xq = dequantize_last_dim(codes, scales) + (x2 - x2.detach())
        y = xq @ w.reshape(self.out_features, self.in_features).T
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
