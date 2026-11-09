"""All-INT8 KS3 cell — full-fidelity port of k-splanifolds ks3_int8.py.

Reference: curvedinf/k-splanifolds experiments/common/ks3_int8.py (line-faithful
math; see RESEARCH.md Sections 8/12). ksdraft deviations, per explicit user
decision ("KS3 quants all of its components very well, so it does not need
FP32 anywhere"):

1. Default nonlinear mode is 'int8' (reference defaults 'fp16').
2. The reference's FP32 exceptions are removed:
   - warp scalars are FP32 buffers in the reference; here they are quantized
     (int8 codes + FP16 scale, one group per chart) and dequantize into the
     arithmetic domain of the active mode.
   - the 'int8' LUT branch's warp intermediate (`scale*(t*(4/127)+shift)` is
     FP32 in the reference) is computed in FP16 here.
   - materialized GEMM outputs / saved coordinates are FP16 (GS scales are
     FP16 per the format; int32 GEMM accumulation is the W8A8 contract).
3. Backward follows RESEARCH.md Section 8: INT8-operand backward GEMMs with
   dynamically requantized gradients, geometry VJP evaluated in FP16, and
   gradients handed off in FP16. Masters stay FP32 (training-system state,
   not the cell's runtime dtype).

Nonlinear modes (copied exactly from the reference `_spine_features`):
- 'fp32'  (NL==0): smooth reference path — parity mode only.
- 'fp16'  (NL==1): staged FP16 rounding; tanh evaluated at FP32 internally
         then rounded (reference behavior, kept).
- 'int8'  (NL==2, default): INT8 tanh LUT (256 entries over +-3.2 domain),
         Q6-guarded signed-INT32 Horner with symmetric `_round_div127`
         division, integer ReLU^2 before requantization.

The packed-coefficient Horner trick is preserved: the spine is evaluated on
int8 CODES and the per-row packed scale `ps` multiplies the summed polynomial
once (exact: the spine is linear in its coefficients).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .quant import GROUP_SIZE, quantize_last_dim, dequantize_last_dim
from .gemm import w8a8_gemm

TANH_DOMAIN = 3.2
FP16 = torch.float16


def make_tanh_lut(device) -> torch.Tensor:
    """256-entry INT8 tanh table over [-3.2, 3.2] (reference construction)."""
    idx = torch.arange(-128, 128, device=device, dtype=torch.float32)
    return (torch.tanh(idx * (TANH_DOMAIN / 127.0)) * 127.0) \
        .round().clamp(-127.0, 127.0).to(torch.int8)


def _round_div127(value: torch.Tensor) -> torch.Tensor:
    """Symmetric round-to-nearest division by 127 on int tensors.

    Triton integer // truncates toward zero; torch.div(trunc) matches, then
    the +-63 remainder correction is applied exactly as the reference.
    """
    quotient = torch.div(value, 127, rounding_mode="trunc")
    remainder = value - quotient * 127
    return quotient + torch.where(remainder > 63, 1,
                                  torch.where(remainder < -63, -1, 0)).to(value.dtype)


def _lut_index(x_fp16: torch.Tensor) -> torch.Tensor:
    """FP16 pre-LUT value -> int32 index into the 256-entry table."""
    scaled = torch.clamp(x_fp16, -TANH_DOMAIN, TANH_DOMAIN) * (127.0 / TANH_DOMAIN)
    return torch.round(scaled).to(torch.int32)


def _grad_handoff(t: torch.Tensor) -> torch.Tensor:
    """FP16 gradient handoff (Section 8), with an FP32 overflow fallback.

    Raw gradient magnitudes can exceed FP16 range before the trainer-side
    loss scaler exists; handing off inf would silently corrupt the masters,
    so oversized tensors fall back to FP32 until that milestone lands."""
    t16 = t.to(FP16)
    if bool(torch.isfinite(t16).all()):
        return t16
    return t.float()


def pack_spine(coefficients: torch.Tensor, group: int = GROUP_SIZE):
    """Pack (charts*rank, 6) coefficient rows; one scale per row (partial
    group), scales flattened — reference pack_spine."""
    rows = coefficients.reshape(-1, 6)
    codes, scales = quantize_last_dim(rows, group)
    return codes, scales.flatten().to(FP16)


class KS3Int8Cell(nn.Module):
    """W8A8 GS-group KS3 with all-INT8 runtime (forward + Section-8 backward).

    Constructed from a torch KS3 cell (`ksdraft.model.ks3.KS3`) whose weights
    are packed; or with ``trainable=True`` to own FP32 masters (reference
    contract) and repack each forward.
    """

    def __init__(self, cell, group: int = GROUP_SIZE, nonlinear: str = "int8",
                 *, trainable: bool = False):
        super().__init__()
        if nonlinear not in ("fp32", "fp16", "int8"):
            raise ValueError("nonlinear must be fp32, fp16 or int8")
        self.group = group
        self.nonlinear = nonlinear
        self.num_subspaces = cell.num_charts
        self.subspace_rank = cell.subspace_rank
        self.in_features = cell.down.weight.shape[1]
        self.out_features = cell.up.weight.shape[0]
        features = self.num_subspaces * self.subspace_rank
        if cell.down.weight.shape[0] != features or cell.up.weight.shape[1] != features:
            raise ValueError("Projection shapes do not match subspace count and rank")

        down = cell.down.weight.detach().float()
        up = cell.up.weight.detach().float()
        coeffs = cell.coefficients.detach().reshape(
            self.num_subspaces, self.subspace_rank, 6)
        raw_scale = cell.raw_scale.detach().float().reshape(self.num_subspaces)
        raw_shift = cell.raw_shift.detach().float().reshape(self.num_subspaces)

        dq, ds = quantize_last_dim(down, group)
        uq, us = quantize_last_dim(up, group)
        self.register_buffer("down_codes", dq)
        self.register_buffer("down_scales", ds)
        self.register_buffer("up_codes", uq)
        self.register_buffer("up_scales", us)
        sq, ss = pack_spine(coeffs, group)
        self.register_buffer("spine_codes", sq)
        self.register_buffer("spine_scale", ss)

        # Warp scalars: transformed then quantized (one group per chart),
        # replacing the reference's FP32 buffers.
        self.register_buffer("warp_scale_raw", raw_scale)   # for backward chains
        self.register_buffer("warp_shift_raw", raw_shift)
        scale_t = F.softplus(raw_scale) + 1e-4
        shift_t = 0.5 * torch.tanh(raw_shift)
        wsc, wss = quantize_last_dim(scale_t.reshape(-1, 1), group)
        whc, whs = quantize_last_dim(shift_t.reshape(-1, 1), group)
        self.register_buffer("warp_scale_codes", wsc)
        self.register_buffer("warp_scale_scales", wss)
        self.register_buffer("warp_shift_codes", whc)
        self.register_buffer("warp_shift_scales", whs)

        self.register_buffer("tanh_lut", make_tanh_lut(down.device))
        self.trainable = trainable
        if trainable:
            for name, value in (("down", down), ("up", up),
                                ("coefficients", coeffs),
                                ("raw_scale", raw_scale), ("raw_shift", raw_shift)):
                setattr(self, "master_" + name, nn.Parameter(value.clone()))

    # ---- scalar dequantization into the active arithmetic domain ----

    def _warp_scalars(self, dtype: torch.dtype):
        scale = dequantize_last_dim(self.warp_scale_codes,
                                    self.warp_scale_scales, self.group) \
            .reshape(-1).to(dtype)
        shift = dequantize_last_dim(self.warp_shift_codes,
                                    self.warp_shift_scales, self.group) \
            .reshape(-1).to(dtype)
        return scale, shift

    # ---- the three nonlinear modes (reference _spine_features) ----

    def _spine_features(self, x, mean, c0, c1, c2, c3, c4, c5, ps, scale, shift):
        """x/mean: (..., charts, rank) coordinates; c*: int8 codes (charts, rank);
        ps: packed row scales (charts,) FP16; scale/shift: (charts,) arithmetic."""
        if self.nonlinear == "fp32":
            # combos computed AFTER leaving int8 (codes span +-127; int8 wraps)
            c0f, c1f, c2f, c3f = (c.float() for c in (c0, c1, c2, c3))
            c4f, c5f = c4.float(), c5.float()
            t = 4.0 * torch.tanh(mean * 0.25)
            z = torch.tanh(scale[:, None] * (t + shift[:, None]))
            a = 2 * c0f - 2 * c1f + c2f + c3f
            b = -3 * c0f + 3 * c1f - 2 * c2f - c3f
            poly = ((a * t + b) * t + c2f) * t + c0f
            warped = ((4 * c5f * z + 2 * c4f) * z - 3 * c5f) * z - c4f
            h = x + (poly + warped) * ps[:, None].float()
            h = torch.relu(h)
            return h * h

        if self.nonlinear == "fp16":
            c0i, c1i, c2i, c3i, c4i, c5i = (c.float() for c in (c0, c1, c2, c3, c4, c5))
            th = (mean * 0.25).to(FP16)
            t = (4.0 * torch.tanh(th.float())).to(FP16)
            warp = (scale[:, None].to(FP16).float()
                    * (t.float() + shift[:, None].to(FP16).float())).to(FP16)
            z = torch.tanh(warp.float()).to(FP16)
            a = (2 * c0i - 2 * c1i + c2i + c3i).to(FP16)
            b = (-3 * c0i + 3 * c1i - 2 * c2i - c3i).to(FP16)
            poly = (a * t + b).to(FP16)
            poly = (poly * t + c2i.to(FP16)).to(FP16)
            poly = (poly * t + c0i.to(FP16)).to(FP16)
            warped = (4 * c5i.to(FP16) * z + 2 * c4i.to(FP16)).to(FP16)
            warped = (warped * z - 3 * c5i.to(FP16)).to(FP16)
            warped = (warped * z - c4i.to(FP16)).to(FP16)
            h = torch.relu((x + (poly.float() + warped.float())
                            * ps[:, None].float()).to(FP16))
            return (h * h).to(FP16).float()

        # 'int8' — reference NL==2, with the warp intermediate in FP16
        ti = _lut_index((mean * 0.25).to(FP16))
        tq = self.tanh_lut[(ti + 128).clamp(0, 255)].to(torch.int32)
        tq16 = tq.to(FP16)
        warp = (scale[:, None].to(FP16) * (tq16 * (4.0 / 127.0)
                                           + shift[:, None].to(FP16))).to(FP16)
        zi = _lut_index(warp)
        zq = self.tanh_lut[(zi + 128).clamp(0, 255)].to(torch.int32)

        c0i, c1i, c2i, c3i, c4i, c5i = (c.to(torch.int32)
                                         for c in (c0, c1, c2, c3, c4, c5))
        a = 2 * c0i - 2 * c1i + c2i + c3i
        b = -3 * c0i + 3 * c1i - 2 * c2i - c3i
        poly = a * 64
        poly = _round_div127(poly * tq * 4) + b * 64
        poly = _round_div127(poly * tq * 4) + c2.to(torch.int32) * 64
        poly = _round_div127(poly * tq * 4) + c0.to(torch.int32) * 64
        warped = 4 * c5.to(torch.int32) * 64
        warped = _round_div127(warped * zq) + 2 * c4.to(torch.int32) * 64
        warped = _round_div127(warped * zq) - 3 * c5.to(torch.int32) * 64
        warped = _round_div127(warped * zq) - c4.to(torch.int32) * 64
        h = x + torch.div(poly + warped, 64, rounding_mode="trunc").to(FP16) \
            * ps[:, None].to(FP16)
        h = torch.relu(h)
        # requantization of the nonnegative ReLU^2 output (codes in [0,127])
        pre_scale = (h.abs().amax(dim=-1, keepdim=True).float() / 127.0) \
            .clamp(2.0**-24, 65504.0).to(FP16).float()
        hq = torch.clamp(torch.round(h / pre_scale), -127.0, 127.0).to(torch.int32)
        hq = torch.clamp(hq, min=0)
        return (hq.to(FP16) * hq.to(FP16)).float() * (pre_scale * pre_scale)

    # ---- forward ----

    def _project(self, x, codes, scales, down_codes, down_scales):
        """Input projection + spine + ReLU^2 + output requantization."""
        M = x.shape[0]
        coords = w8a8_gemm(x, scales, down_codes, down_scales, self.group) \
            .to(FP16).clamp(-65504.0, 65504.0).float()  # materialize + clamp
        coords = coords.reshape(M, self.num_subspaces, self.subspace_rank)
        mean = coords if self.subspace_rank == 1 else coords.mean(-1, keepdim=True)

        c_all = self.spine_codes.reshape(self.num_subspaces * self.subspace_rank, 6)
        c0, c1, c2, c3, c4, c5 = c_all.unbind(-1)
        c0 = c0.reshape(self.num_subspaces, self.subspace_rank)
        c1 = c1.reshape(self.num_subspaces, self.subspace_rank)
        c2 = c2.reshape(self.num_subspaces, self.subspace_rank)
        c3 = c3.reshape(self.num_subspaces, self.subspace_rank)
        c4 = c4.reshape(self.num_subspaces, self.subspace_rank)
        c5 = c5.reshape(self.num_subspaces, self.subspace_rank)
        ps = self.spine_scale.reshape(self.num_subspaces)
        arith = torch.float32 if self.nonlinear == "fp32" else FP16
        scale, shift = self._warp_scalars(arith)

        h = self._spine_features(coords, mean, c0, c1, c2, c3, c4, c5,
                                 ps, scale, shift)
        return h.reshape(M, -1), coords

    @torch.no_grad()
    def _repack_masters(self):
        dq, ds = quantize_last_dim(self.master_down, self.group)
        uq, us = quantize_last_dim(self.master_up, self.group)
        sq, ss = pack_spine(self.master_coefficients, self.group)
        self.down_codes, self.down_scales = dq, ds
        self.up_codes, self.up_scales = uq, us
        self.spine_codes, self.spine_scale = sq, ss
        scale_t = F.softplus(self.master_raw_scale.detach()) + 1e-4
        shift_t = 0.5 * torch.tanh(self.master_raw_shift.detach())
        wsc, wss = quantize_last_dim(scale_t.reshape(-1, 1), self.group)
        whc, whs = quantize_last_dim(shift_t.reshape(-1, 1), self.group)
        self.warp_scale_codes, self.warp_scale_scales = wsc, wss
        self.warp_shift_codes, self.warp_shift_scales = whc, whs
        self.warp_scale_raw = self.master_raw_scale.detach()
        self.warp_shift_raw = self.master_raw_shift.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            return _KS3Int8Train.apply(
                x, self.master_down, self.master_up, self.master_coefficients,
                self.master_raw_scale, self.master_raw_shift, self)
        x2 = x.reshape(-1, x.shape[-1])
        xq, xs = quantize_last_dim(x2.float(), self.group)
        h, _ = self._project(xq, xq, xs, self.down_codes, self.down_scales)
        hq, hs = quantize_last_dim(h, self.group)
        y = w8a8_gemm(hq, hs, self.up_codes, self.up_scales, self.group)
        out_dtype = (torch.get_autocast_gpu_dtype()
                     if torch.is_autocast_enabled("cuda") else x.dtype)
        return y.reshape(*x.shape[:-1], self.out_features).to(out_dtype)


class _KS3Int8Train(torch.autograd.Function):
    """Section-8 training forward/backward: all-INT8 runtime, FP16 handoff.

    Backward = reference geometry VJP (exact eight-partial form, FP16
    arithmetic) composed with INT8 backward GEMMs on dynamically requantized
    gradients. Returned gradients are FP16.
    """

    @staticmethod
    def forward(ctx, inputs, down, up, coefficients, raw_scale, raw_shift, cell):
        cell._repack_masters()
        x = inputs.reshape(-1, inputs.shape[-1]).float()
        xq, xs = quantize_last_dim(x, cell.group)
        h, coords = cell._project(xq, xq, xs, cell.down_codes, cell.down_scales)
        hq, hs = quantize_last_dim(h, cell.group)
        y = w8a8_gemm(hq, hs, cell.up_codes, cell.up_scales, cell.group)
        ctx.cell = cell
        ctx.input_shape = inputs.shape
        ctx.save_for_backward(xq, xs, hq, hs, coords)
        out_dtype = (torch.get_autocast_gpu_dtype()
                     if torch.is_autocast_enabled("cuda") else inputs.dtype)
        return y.reshape(*inputs.shape[:-1], cell.out_features).to(out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        cell = ctx.cell
        xq, xs, hq, hs, coords = ctx.saved_tensors
        M, features = coords.shape[:2][0], hq.shape[1]
        g = grad_out.reshape(-1, cell.out_features).float().to(FP16)

        # output-projection backward GEMM (INT8 operands): d(coords-pre-relu)
        gq, gs = quantize_last_dim(g.float(), cell.group)
        up_real = dequantize_last_dim(cell.up_codes, cell.up_scales, cell.group)
        upT_q, upT_s = quantize_last_dim(
            up_real.t().contiguous(), cell.group)
        value = w8a8_gemm(gq, gs, upT_q, upT_s, cell.group).to(FP16)

        # ---- geometry VJP (reference form, FP16) ----
        c_all = cell.spine_codes.reshape(-1, 6).float() \
            * cell.spine_scale.reshape(-1, 1).float()
        c0 = c_all[:, 0].reshape(cell.num_subspaces, cell.subspace_rank).to(FP16)
        c1 = c_all[:, 1].reshape(cell.num_subspaces, cell.subspace_rank).to(FP16)
        c2 = c_all[:, 2].reshape(cell.num_subspaces, cell.subspace_rank).to(FP16)
        c3 = c_all[:, 3].reshape(cell.num_subspaces, cell.subspace_rank).to(FP16)
        c4 = c_all[:, 4].reshape(cell.num_subspaces, cell.subspace_rank).to(FP16)
        c5 = c_all[:, 5].reshape(cell.num_subspaces, cell.subspace_rank).to(FP16)
        scale, shift = cell._warp_scalars(FP16)
        scale, shift = scale.reshape(1, -1, 1), shift.reshape(1, -1, 1)
        x16 = coords.to(FP16)
        t = 4.0 * torch.tanh(x16 * 0.25)
        z = torch.tanh(scale * (t + shift))
        cubic = 2 * c0 - 2 * c1 + c2 + c3
        quad = -3 * c0 + 3 * c1 - 2 * c2 - c3
        spine = ((cubic * t + quad) * t + c2) * t + c0
        warped = ((4 * c5 * z + 2 * c4) * z - 3 * c5) * z - c4
        dpre = value.reshape(M, cell.num_subspaces, cell.subspace_rank) \
            * 2.0 * torch.relu(x16 + spine + warped)
        dpoly = (3 * cubic * t + 2 * quad) * t + c2
        dwarp = (12 * c5 * z + 4 * c4) * z - 3 * c5
        dz = dwarp * (1.0 - z * z)
        dmean = dpre * (dpoly + dz * scale) * (1.0 - t * t * 0.0625)

        t2, t3 = t * t, t * t * t
        z2, z3 = z * z, z * z * z
        d0 = (dpre * (2 * t3 - 3 * t2 + 1)).sum(0)
        d1 = (dpre * (-2 * t3 + 3 * t2)).sum(0)
        d2 = (dpre * (t3 - 2 * t2 + t)).sum(0)
        d3 = (dpre * (t3 - t2)).sum(0)
        d4 = (dpre * (2 * z2 - 1)).sum(0)
        d5 = (dpre * (4 * z3 - 3 * z)).sum(0)
        dscale_t = (dpre * dz * (t + shift)).reshape(M, -1).sum(0)
        dshift_t = (dpre * dz * scale).reshape(M, -1).sum(0) * 0.5
        dcoeff = torch.stack((d0, d1, d2, d3, d4, d5), -1) \
            .reshape(cell.num_subspaces, cell.subspace_rank, 6).to(FP16)
        # STE through the transformed-scalar quantization back to raw params
        sig = torch.sigmoid(cell.warp_scale_raw.float())
        th = torch.tanh(cell.warp_shift_raw.float())
        dscale = (dscale_t.float() * sig).to(FP16)
        dshift = (dshift_t.float() * (1 - th * th)).to(FP16)

        # coordinate gradient (rank-1: direct + shared; general: mean share)
        dc = value.reshape(M, cell.num_subspaces, cell.subspace_rank) + dmean
        if cell.subspace_rank > 1:
            dc = dc + dmean.sum(-1, keepdim=True) / cell.subspace_rank
        dc = dc.reshape(M, -1).float()

        # ---- INT8 backward GEMMs (dynamically requantized transposes) ----
        # Operands are built from DEQUANTIZED real values (codes are +-127
        # integers; requantizing codes directly would scale the GEMM by the
        # code magnitude instead of the weight magnitude).
        def _tq(t):  # transpose + requantize real values for the (N, K) layout
            return quantize_last_dim(t.t().contiguous(), cell.group)

        dcq, dcs = quantize_last_dim(dc, cell.group)
        down_real = dequantize_last_dim(cell.down_codes, cell.down_scales, cell.group)
        dx = w8a8_gemm(dcq, dcs, *_tq(down_real), cell.group)
        # ddown needs one operand quantized along the short batch dim (the
        # (N,K) kernel layout forces it); measured cosine vs the fp32
        # reference backward is ~0.82 — flagged as a float-exception
        # candidate for the acceptance gates.
        x_real = dequantize_last_dim(xq, xs, cell.group)
        dcT_q, dcT_s = quantize_last_dim(dc.t().contiguous(), cell.group)
        xT_q, xT_s = quantize_last_dim(x_real.t().contiguous(), cell.group)
        ddown = w8a8_gemm(dcT_q, dcT_s, xT_q, xT_s, cell.group)
        up_real = dequantize_last_dim(cell.up_codes, cell.up_scales, cell.group)
        gT_q, gT_s = quantize_last_dim(g.float().t().contiguous(), cell.group)
        h_real = dequantize_last_dim(hq, hs, cell.group)
        dup = w8a8_gemm(gT_q, gT_s, *_tq(h_real), cell.group)

        dx = dx.reshape(ctx.input_shape)
        return (dx.to(grad_out.dtype), _grad_handoff(ddown), _grad_handoff(dup),
                _grad_handoff(dcoeff), _grad_handoff(dscale), _grad_handoff(dshift),
                None)
