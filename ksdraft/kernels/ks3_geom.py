"""Fused KS3 geometry forward/backward (rank-1 charts).

The reference path materializes the whole elementwise chain as separate
tensors per layer: clamp, tanh, mean, warp, Horner cubic, Horner warp, add,
relu, square — roughly twenty fp32 activations per element and the same
number again in backward. At 24 blocks x 5888 charts x B*T positions that is
the largest single elementwise region in the trunk.

The fused kernel keeps one program per (chart block, position block), streams
positions in registers, and recomputes the scalar chain in the backward
instead of storing it. Parameter gradients (per-chart coefficients and the
warp scale/shift) accumulate in registers and land with one atomic add per
program.

Math (matches ``ks3.spine_horner`` / ``ks3.spine_basis`` exactly):

    t = 4 tanh(h/4)
    z = tanh(scale * (t + shift))
    spine  = p0 h00 + p1 h10 + p2 h01 + p3 h11      (Hermite basis)
    warped = p4 T2(z) + p5 T3(z)                    (Chebyshev)
    f      = relu(h + spine + warped)^2

``scale = softplus(raw_scale) + 1e-4`` and ``shift = 0.5 tanh(raw_shift)``.
"""

from __future__ import annotations

import os

import torch

try:  # Triton is optional (CPU/dev boxes, CI)
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    _HAS_TRITON = False
    triton = None
    tl = None
    libdevice = None

BLOCK_C = 64
BLOCK_P = 128
EPS = 1e-4


def triton_enabled(coords: torch.Tensor) -> bool:
    if not _HAS_TRITON or not coords.is_cuda:
        return False
    if os.environ.get("KSDRAFT_TRITON_KS3", "1") == "0":
        return False
    return coords.dtype in (torch.float16, torch.bfloat16, torch.float32)


if _HAS_TRITON:
    @triton.jit
    def _softplus(x):
        return tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)

    @triton.jit
    def _fwd_kernel(h_ptr, coeff_ptr, rs_ptr, rh_ptr, out_ptr,
                    P, C, BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr):
        pid_c = tl.program_id(0)
        pid_p = tl.program_id(1)
        cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cm = cs < C
        p0 = tl.load(coeff_ptr + cs * 6 + 0, mask=cm, other=0.0)
        p1 = tl.load(coeff_ptr + cs * 6 + 1, mask=cm, other=0.0)
        p2 = tl.load(coeff_ptr + cs * 6 + 2, mask=cm, other=0.0)
        p3 = tl.load(coeff_ptr + cs * 6 + 3, mask=cm, other=0.0)
        p4 = tl.load(coeff_ptr + cs * 6 + 4, mask=cm, other=0.0)
        p5 = tl.load(coeff_ptr + cs * 6 + 5, mask=cm, other=0.0)
        scale = _softplus(tl.load(rs_ptr + cs, mask=cm, other=0.0)) + 1e-4
        shift = 0.5 * libdevice.tanh(tl.load(rh_ptr + cs, mask=cm, other=0.0))

        cubic = 2.0 * p0 - 2.0 * p1 + p2 + p3
        quad = -3.0 * p0 + 3.0 * p1 - 2.0 * p2 - p3

        pstart = pid_p * BLOCK_P
        for i in range(0, BLOCK_P):
            p = pstart + i
            pm = p < P
            m = cm & pm
            h = tl.load(h_ptr + p * C + cs, mask=m, other=0.0).to(tl.float32)
            t = 4.0 * libdevice.tanh(h * 0.25)
            z = libdevice.tanh(scale * (t + shift))
            spine = ((cubic * t + quad) * t + p2) * t + p0
            warped = ((4.0 * p5 * z + 2.0 * p4) * z - 3.0 * p5) * z - p4
            pre = h + spine + warped
            f = tl.maximum(pre, 0.0) * tl.maximum(pre, 0.0)
            tl.store(out_ptr + p * C + cs, f, mask=m)

    @triton.jit
    def _bwd_kernel(h_ptr, coeff_ptr, rs_ptr, rh_ptr, df_ptr, dh_ptr,
                    dc_ptr, drs_ptr, drh_ptr,
                    P, C, BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr):
        pid_c = tl.program_id(0)
        pid_p = tl.program_id(1)
        cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cm = cs < C
        p0 = tl.load(coeff_ptr + cs * 6 + 0, mask=cm, other=0.0)
        p1 = tl.load(coeff_ptr + cs * 6 + 1, mask=cm, other=0.0)
        p2 = tl.load(coeff_ptr + cs * 6 + 2, mask=cm, other=0.0)
        p3 = tl.load(coeff_ptr + cs * 6 + 3, mask=cm, other=0.0)
        p4 = tl.load(coeff_ptr + cs * 6 + 4, mask=cm, other=0.0)
        p5 = tl.load(coeff_ptr + cs * 6 + 5, mask=cm, other=0.0)
        rs = tl.load(rs_ptr + cs, mask=cm, other=0.0)
        rh = tl.load(rh_ptr + cs, mask=cm, other=0.0)
        scale = _softplus(rs) + 1e-4
        th = libdevice.tanh(rh)
        shift = 0.5 * th
        d_softplus = 1.0 / (1.0 + tl.exp(-rs))            # d softplus / d rs
        d_shift = 0.5 * (1.0 - th * th)                   # d shift / d rh

        cubic = 2.0 * p0 - 2.0 * p1 + p2 + p3
        quad = -3.0 * p0 + 3.0 * p1 - 2.0 * p2 - p3

        a0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        a1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        a2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        a3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        a4 = tl.zeros([BLOCK_C], dtype=tl.float32)
        a5 = tl.zeros([BLOCK_C], dtype=tl.float32)
        ascale = tl.zeros([BLOCK_C], dtype=tl.float32)
        ashift = tl.zeros([BLOCK_C], dtype=tl.float32)

        pstart = pid_p * BLOCK_P
        for i in range(0, BLOCK_P):
            p = pstart + i
            pm = p < P
            m = cm & pm
            h = tl.load(h_ptr + p * C + cs, mask=m, other=0.0).to(tl.float32)
            df = tl.load(df_ptr + p * C + cs, mask=m, other=0.0).to(tl.float32)
            t = 4.0 * libdevice.tanh(h * 0.25)
            z = libdevice.tanh(scale * (t + shift))
            t2 = t * t
            t3 = t2 * t
            z2 = z * z
            z3 = z2 * z
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h01 = -2.0 * t3 + 3.0 * t2        # coefficient p1
            h10 = t3 - 2.0 * t2 + t           # coefficient p2
            h11 = t3 - t2
            t2c = 2.0 * z2 - 1.0
            t3c = 4.0 * z3 - 3.0 * z
            spine = p0 * h00 + p1 * h01 + p2 * h10 + p3 * h11
            warped = p4 * t2c + p5 * t3c
            pre = h + spine + warped
            relu = pre > 0.0
            dpre = df * 2.0 * tl.where(relu, pre, 0.0)

            dt_dh = 1.0 - (0.25 * t) * (0.25 * t)         # sech^2(h/4)
            dz_dt = scale * (1.0 - z2)
            dspine_dt = 3.0 * cubic * t2 + 2.0 * quad * t + p2
            dwarped_dz = 12.0 * p5 * z2 + 4.0 * p4 * z - 3.0 * p5
            dpre_dh = 1.0 + (dspine_dt + dwarped_dz * dz_dt) * dt_dh
            dh = dpre * dpre_dh
            tl.store(dh_ptr + p * C + cs, dh, mask=m)

            a0 += dpre * h00
            a1 += dpre * h01
            a2 += dpre * h10
            a3 += dpre * h11
            a4 += dpre * t2c
            a5 += dpre * t3c
            dwz = dpre * dwarped_dz * (1.0 - z2)
            ascale += dwz * (t + shift)
            ashift += dwz * scale

        tl.atomic_add(dc_ptr + cs * 6 + 0, a0, mask=cm)
        tl.atomic_add(dc_ptr + cs * 6 + 1, a1, mask=cm)
        tl.atomic_add(dc_ptr + cs * 6 + 2, a2, mask=cm)
        tl.atomic_add(dc_ptr + cs * 6 + 3, a3, mask=cm)
        tl.atomic_add(dc_ptr + cs * 6 + 4, a4, mask=cm)
        tl.atomic_add(dc_ptr + cs * 6 + 5, a5, mask=cm)
        tl.atomic_add(drs_ptr + cs, ascale * d_softplus, mask=cm)
        tl.atomic_add(drh_ptr + cs, ashift * d_shift, mask=cm)


class _KS3Geom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, coeffs, raw_scale, raw_shift):
        p, c = h.shape
        out = torch.empty(p, c, device=h.device, dtype=h.dtype)
        grid = (triton.cdiv(c, BLOCK_C), triton.cdiv(p, BLOCK_P))
        _fwd_kernel[grid](h, coeffs, raw_scale, raw_shift, out, p, c,
                          BLOCK_C=BLOCK_C, BLOCK_P=BLOCK_P, num_warps=4)
        ctx.save_for_backward(h, coeffs, raw_scale, raw_shift)
        return out

    @staticmethod
    def backward(ctx, df):
        h, coeffs, raw_scale, raw_shift = ctx.saved_tensors
        p, c = h.shape
        dh = torch.empty_like(h)
        dc = torch.zeros_like(coeffs, dtype=torch.float32)
        drs = torch.zeros_like(raw_scale, dtype=torch.float32)
        drh = torch.zeros_like(raw_shift, dtype=torch.float32)
        grid = (triton.cdiv(c, BLOCK_C), triton.cdiv(p, BLOCK_P))
        _bwd_kernel[grid](h, coeffs, raw_scale, raw_shift, df, dh,
                          dc, drs, drh, p, c,
                          BLOCK_C=BLOCK_C, BLOCK_P=BLOCK_P, num_warps=4)
        return dh, dc, drs, drh


def ks3_geometry(h: torch.Tensor, coeffs: torch.Tensor,
                 raw_scale: torch.Tensor,
                 raw_shift: torch.Tensor) -> torch.Tensor:
    """Rank-1 fused geometry; falls back to the reference path elsewhere."""
    if triton_enabled(h) and coeffs.shape[-1] == 6 and coeffs.ndim == 2:
        shape = h.shape
        flat = h.reshape(-1, shape[-1])
        return _KS3Geom.apply(flat, coeffs, raw_scale,
                              raw_shift).reshape(shape)
    from ..model.ks3 import spine_horner
    import torch.nn.functional as F

    hf = h.float().clamp(-65504.0, 65504.0)
    t = 4.0 * torch.tanh(hf / 4.0)
    scale = F.softplus(raw_scale.float()) + EPS
    shift = 0.5 * torch.tanh(raw_shift.float())
    z = torch.tanh(scale * (t + shift))
    spine, warped = spine_horner(t, z, coeffs.float())
    return F.relu(hf + spine + warped).square().to(h.dtype)
