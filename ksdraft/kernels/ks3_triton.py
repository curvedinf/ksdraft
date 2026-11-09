"""Fused Triton kernels for the all-INT8 KS3 cell — Horner-style impl.

Line-faithful port of the reference fused kernels (k-splanifolds
experiments/common/ks3_int8.py: `_gemm` FUSED path, `_spine_features`,
`_half_scale`, `_round_div127`, `_quant`), generalized from the hard-wired
GS32 to a constexpr GROUP (ksdraft default 128; 32 = reference parity grid).

The input-projection GEMM evaluates the packed-coefficient Horner spine in
its epilogue and requantizes the ReLU^2 features directly (int8 codes +
fp16 scales) for the output-projection GEMM — one kernel launch instead of
three, exactly the reference design ("BN==group" so each program writes one
quantization group).

ksdraft deviations from the reference arithmetic (user-directed, both
documented and parity-tested in tests/test_ks3_int8.py):
- warp scalars are quantized; the kernel loads them as fp16 (reference:
  fp32 buffers).
- NL==2 (int8 LUT) computes the warp intermediate in fp16 (reference: fp32).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP16 = torch.float16
TANH_DOMAIN = 3.2


@triton.jit
def _half_scale(maximum):
    # Round BEFORE quantizing so encoding and decoding use the same FP16 scale.
    return tl.minimum(tl.maximum(maximum / 127., 2.**-24), 65504.).to(tl.float16).to(tl.float32)


@triton.jit
def _round_div127(value):
    # Triton integer // truncates toward zero. Correct both signs explicitly.
    quotient = value // 127
    remainder = value - quotient * 127
    return quotient + tl.where(remainder > 63, 1, tl.where(remainder < -63, -1, 0))


@triton.jit
def _spine_features(x, mean, c0, c1, c2, c3, c4, c5, ps, scale, shift, LUT,
                    valid, NL: tl.constexpr):
    """Reference `_spine_features`, verbatim math. c0..c5 are int8 CODES;
    ps is the packed per-row scale applied once (Horner on codes)."""
    a = 2 * c0 - 2 * c1 + c2 + c3
    b = -3 * c0 + 3 * c1 - 2 * c2 - c3
    if NL == 2:
        # INT8 tanh inputs/outputs, INT32 fixed-point Horner accumulators.
        # Q6 guard bits keep worst-case polynomial products within INT32.
        ti = libdevice.nearbyint(tl.minimum(tl.maximum(mean * .25, -3.2), 3.2) * (127. / 3.2)).to(tl.int32)
        tq = tl.load(LUT + ti + 128).to(tl.int32)
        # ksdraft deviation: warp intermediate in FP16 (reference: FP32)
        warp = (scale.to(tl.float16) * (tq.to(tl.float16) * (4. / 127.) + shift.to(tl.float16))).to(tl.float32)
        zi = libdevice.nearbyint(tl.minimum(tl.maximum(warp, -3.2), 3.2) * (127. / 3.2)).to(tl.int32)
        zq = tl.load(LUT + zi + 128).to(tl.int32)
        poly = a[None, :].to(tl.int32) * 64
        poly = _round_div127(poly * tq * 4) + b[None, :].to(tl.int32) * 64
        poly = _round_div127(poly * tq * 4) + c2[None, :].to(tl.int32) * 64
        poly = _round_div127(poly * tq * 4) + c0[None, :].to(tl.int32) * 64
        warped = 4 * c5[None, :].to(tl.int32) * 64
        warped = _round_div127(warped * zq) + 2 * c4[None, :].to(tl.int32) * 64
        warped = _round_div127(warped * zq) - 3 * c5[None, :].to(tl.int32) * 64
        warped = _round_div127(warped * zq) - c4[None, :].to(tl.int32) * 64
        h = x + (poly + warped).to(tl.float32) * (ps[None, :] / 64.)
        h = tl.where(valid[None, :], h, 0.)
        pre_scale = _half_scale(tl.max(tl.abs(h), 1))
        hq = tl.minimum(tl.maximum(libdevice.nearbyint(h / pre_scale[:, None]), -127.), 127.).to(tl.int32)
        hq = tl.maximum(hq, 0)
        h = (hq * hq).to(tl.float32) * (pre_scale[:, None] * pre_scale[:, None])
    elif NL == 1:
        th = (mean * .25).to(tl.float16)
        t = (4 * libdevice.tanh(th.to(tl.float32))).to(tl.float16)
        warp = (scale[None, :] * (t.to(tl.float32) + shift[None, :])).to(tl.float16)
        z = libdevice.tanh(warp.to(tl.float32)).to(tl.float16)
        # Each polynomial stage rounds to FP16; tanh uses FP32 internally.
        poly = (a[None, :].to(tl.float16) * t + b[None, :].to(tl.float16)).to(tl.float16)
        poly = (poly * t + c2[None, :].to(tl.float16)).to(tl.float16)
        poly = (poly * t + c0[None, :].to(tl.float16)).to(tl.float16)
        warped = (4 * c5[None, :].to(tl.float16) * z + 2 * c4[None, :].to(tl.float16)).to(tl.float16)
        warped = (warped * z - 3 * c5[None, :].to(tl.float16)).to(tl.float16)
        warped = (warped * z - c4[None, :].to(tl.float16)).to(tl.float16)
        h = tl.maximum((x + (poly.to(tl.float32) + warped.to(tl.float32)) * ps[None, :]).to(tl.float16), 0.)
        h = (h * h).to(tl.float16).to(tl.float32)
    else:
        t = 4 * libdevice.tanh(mean * .25)
        z = libdevice.tanh(scale[None, :] * (t + shift[None, :]))
        poly = ((a[None, :] * t + b[None, :]) * t + c2[None, :]) * t + c0[None, :]
        warped = ((4 * c5[None, :] * z + 2 * c4[None, :]) * z - 3 * c5[None, :]) * z - c4[None, :]
        h = tl.maximum(x + (poly + warped) * ps[None, :], 0.)
        h = h * h
    return tl.where(valid[None, :], h, 0.)


@triton.jit
def _fused_gemm_spine(
    A, AS, W, WS,          # input codes/scales (M, Kp), weight codes/scales (N, Kp)
    H, HS,                 # out: feature codes (M, Np), scales (M, Np/G)
    P, PS,                 # packed spine codes (N, 6), flattened row scales (N,)
    SC, SH,                # warp scale/shift (fp16, per chart = per N row)
    LUT,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    NSCALE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    NL: tl.constexpr,
):
    """Input projection GEMM + Horner spine + ReLU^2 + requant, fused.

    Rank-1 path of the reference `_gemm` with FUSED=True: each program
    computes one BM x BN tile; BN == G so the epilogue writes exactly one
    quantization group per program row-block. `mean` is the coordinate
    itself (rank 1)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    mi = pid_m * BM + tl.arange(0, BM)
    ni = pid_n * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    NG = K // G

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for group in range(NG):
        kk = group * G + ki
        av = tl.load(A + mi[:, None] * K + kk[None, :],
                     (mi[:, None] < M) & (kk[None, :] < K), other=0)
        wv = tl.load(W + ni[None, :] * K + kk[:, None],
                     (ni[None, :] < N) & (kk[:, None] < K), other=0)
        dot = tl.dot(av, wv, out_dtype=tl.int32)
        sa = tl.load(AS + mi * NG + group, mi < M, other=0.).to(tl.float32)
        sw = tl.load(WS + ni * NG + group, ni < N, other=0.).to(tl.float32)
        acc += dot.to(tl.float32) * sa[:, None] * sw[None, :]

    valid = ni < N
    x = tl.minimum(tl.maximum(acc, -65504.), 65504.)
    # codes cast at load (reference): the 2*c0-2*c1+... combos wrap in int8
    c0 = tl.load(P + ni * 6 + 0, valid, other=0).to(tl.float32)
    c1 = tl.load(P + ni * 6 + 1, valid, other=0).to(tl.float32)
    c2 = tl.load(P + ni * 6 + 2, valid, other=0).to(tl.float32)
    c3 = tl.load(P + ni * 6 + 3, valid, other=0).to(tl.float32)
    c4 = tl.load(P + ni * 6 + 4, valid, other=0).to(tl.float32)
    c5 = tl.load(P + ni * 6 + 5, valid, other=0).to(tl.float32)
    ps = tl.load(PS + ni, valid, other=0.).to(tl.float32)
    scale = tl.load(SC + ni, valid, other=0.)
    shift = tl.load(SH + ni, valid, other=0.)
    h = _spine_features(x, x, c0, c1, c2, c3, c4, c5, ps, scale, shift, LUT,
                        valid, NL)
    h = tl.where(valid[None, :], h, 0.)
    hs = _half_scale(tl.max(tl.abs(h), 1))
    q = tl.minimum(tl.maximum(libdevice.nearbyint(h / hs[:, None]), -127.), 127.).to(tl.int8)
    tl.store(H + mi[:, None] * N + ni[None, :], q,
             (mi[:, None] < M) & valid[None, :])
    tl.store(HS + mi * NSCALE + pid_n, hs, mi < M)


def project_fused(
    a_codes: torch.Tensor, a_scales: torch.Tensor,
    w_codes: torch.Tensor, w_scales: torch.Tensor,
    spine_codes: torch.Tensor, spine_scale: torch.Tensor,
    warp_scale: torch.Tensor, warp_shift: torch.Tensor,
    tanh_lut: torch.Tensor,
    group: int, nonlinear: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused input projection + Horner spine + requant on CUDA.

    warp_scale/warp_shift: DEQUANTIZED fp16 per-chart scalars.
    Returns (h_codes (M, N), h_scales (M, ceil(N/group))) ready for the
    output-projection W8A8 GEMM."""
    M, K = a_codes.shape
    N = w_codes.shape[0]
    Kp = (K + group - 1) // group * group
    if Kp != K:
        a_codes = torch.nn.functional.pad(a_codes, (0, Kp - K))
        w_codes = torch.nn.functional.pad(w_codes, (0, Kp - K))
        K = Kp
    Np = (N + group - 1) // group * group

    h = torch.zeros((M, N), device=a_codes.device, dtype=torch.int8)
    hs = torch.zeros((M, Np // group), device=a_codes.device, dtype=FP16)
    BM, BN = 16, group
    BK = group  # one full quantization group per K step (reference: BK == G)
    if group & (group - 1):
        raise ValueError("group must be a power of two (32 or 128)")
    nl = {"fp32": 0, "fp16": 1, "int8": 2}[nonlinear]
    nscale = Np // group
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fused_gemm_spine[grid](
        a_codes, a_scales, w_codes, w_scales, h, hs,
        spine_codes, spine_scale,
        warp_scale.to(FP16).contiguous(), warp_shift.to(FP16).contiguous(),
        tanh_lut, M, N, K, group, nscale, BM, BN, BK, nl, num_warps=4)
    return h, hs
