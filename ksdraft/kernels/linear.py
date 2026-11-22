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
        self._maybe_quantize_weight()

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
        dither = float(getattr(self, "_dither", 0.0))
        if dither > 0:
            from ..train.stages import quantize_last_dim_dithered
            codes, scales = quantize_last_dim_dithered(
                self.weight.detach().float(), 128, dither)
        else:
            codes, scales = quantize_last_dim(self.weight.detach().float())
        self.weight_codes = codes
        self.weight_scales = scales
        self._codes_ver = None if self.weight.is_inference() else self.weight._version
        self._codes_dither = dither

    def _maybe_quantize_weight(self) -> None:
        """Re-quantize only when the fp32 masters changed.

        The eval/proposal path must not repack the (up to 463M-weight)
        vocabulary head or any other surface on every forward; a version
        check keeps codes live exactly while masters move.
        """
        if (self.weight_codes is None
                or self.weight.is_inference()
                or self._codes_ver != self.weight._version
                or self._codes_dither != getattr(self, "_dither", 0.0)):
            self._quantize_weight()

    def _quantize_weight_dithered(self, dither: float) -> None:
        self._dither = dither
        self._quantize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.weight.requires_grad:
            # training: real INT8 forward (INT8 operands, INT32 accumulate,
            # fp32 epilogue); gradients hand off in fp32 through the STE
            return _W8A8Linear.apply(
                x, self.weight, self.bias,
                float(getattr(self, "_dither", 0.0)))
        self._maybe_quantize_weight()  # repacks only when masters changed
        x2 = x.reshape(-1, x.shape[-1])
        a_codes, a_scales = quantize_last_dim(x2.float())
        y = w8a8_gemm(a_codes, a_scales, self.weight_codes, self.weight_scales)
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

    def forward_codes(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """INT8 input path: codes+scales are the cache form (g128 groups).

        Skips activation quantization — the operand already *is* the int8
        cache content; the group scales ride straight into the GEMM epilogue.
        Gradients (training) flow to the weight/bias through the STE.
        """
        if torch.is_grad_enabled() and self.weight.requires_grad:
            return _W8A8LinearCodes.apply(codes, scales, self.weight, self.bias)
        self._maybe_quantize_weight()
        y = w8a8_gemm(codes.reshape(-1, codes.shape[-1]),
                      scales.reshape(-1, scales.shape[-1]).float(),
                      self.weight_codes, self.weight_scales)
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*codes.shape[:-1], self.out_features)

    def fake_quant_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STE path: fp32 GEMM on quantize-dequantized operands (autograd-safe).
        Honors ``self._dither`` (PTQR annealing) if set."""
        from .quant import dequantize_last_dim
        from . import quant as _q

        dither = float(getattr(self, "_dither", 0.0))
        if dither > 0:
            from . import gemm as _g  # noqa: F401
            from ..train.stages import quantize_last_dim_dithered
            self._quantize_weight_dithered(dither)
        else:
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


class _W8A8LinearCodes(torch.autograd.Function):
    """INT8-code input (cache form): codes (..., K), scales (..., K/128).

    Forward runs the W8A8 GEMM with the cache's own scales; the input is
    int8 data (no gradient), the weight/bias grads come back through the
    dequantized-weight STE in fp32.
    """

    @staticmethod
    def forward(ctx, codes, scales, w_master, bias):
        w_codes, w_scales = quantize_last_dim(w_master.detach().float())
        y = w8a8_gemm(codes.reshape(-1, codes.shape[-1]),
                      scales.reshape(-1, scales.shape[-1]).float(),
                      w_codes, w_scales)
        if bias is not None:
            y = y + bias.float()
        ctx.save_for_backward(codes, scales, w_codes, w_scales)
        ctx.has_bias = bias is not None
        ctx.out_shape = (*codes.shape[:-1], y.shape[-1])
        return y.reshape(ctx.out_shape)

    @staticmethod
    def backward(ctx, grad_out):
        codes, scales, w_codes, w_scales = ctx.saved_tensors
        from .quant import dequantize_last_dim

        xq = dequantize_last_dim(codes.float(), scales).reshape(
            -1, codes.shape[-1])
        g = grad_out.float().reshape(-1, grad_out.shape[-1])
        dw = g.t() @ xq
        db = g.sum(0) if ctx.has_bias else None
        return None, None, dw, db


class _W8A8Linear(torch.autograd.Function):
    """Training forward: INT8 operands, INT32 accumulation, scaled epilogue.

    Forward quantizes activations per token (g128) and weights per output row
    (g128) and runs the W8A8 GEMM. Backward is the standard STE: gradients
    flow through dequantized operands in fp32 (no half precision anywhere).
    """

    @staticmethod
    def forward(ctx, x, w_master, bias, dither):
        x2 = x.reshape(-1, x.shape[-1]).float()
        if dither > 0:
            from ..train.stages import quantize_last_dim_dithered
            w_codes, w_scales = quantize_last_dim_dithered(
                w_master.detach().float(), 128, dither)
        else:
            w_codes, w_scales = quantize_last_dim(w_master.detach().float())
        a_codes, a_scales = quantize_last_dim(x2)
        y = w8a8_gemm(a_codes, a_scales, w_codes, w_scales)
        if bias is not None:
            y = y + bias.float()
        ctx.save_for_backward(a_codes, a_scales, w_codes, w_scales)
        ctx.has_bias = bias is not None
        ctx.in_shape = x.shape
        return y.reshape(*x.shape[:-1], y.shape[-1]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        a_codes, a_scales, w_codes, w_scales = ctx.saved_tensors
        from .quant import dequantize_last_dim

        w = dequantize_last_dim(w_codes.float(), w_scales)
        xq = dequantize_last_dim(a_codes.float(), a_scales)
        g = grad_out.float().reshape(-1, grad_out.shape[-1])
        dx = (g @ w).reshape(ctx.in_shape)
        dw = g.t() @ xq
        db = g.sum(0) if ctx.has_bias else None
        return dx, dw, db, None
