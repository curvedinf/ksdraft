"""W8A8 GS128 GEMM.

Triton kernel: INT8 operands, INT32 dot accumulation (tl.dot), per-group
FP16 scales promoted to FP32 for the epilogue, one 128-lane group per K step.
A pure-torch fallback (dequantize then fp32 matmul) covers CPU and any device
without Triton, and doubles as the kernel's parity reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .quant import GROUP_SIZE, SCALE_DTYPE


@triton.jit
def _w8a8_gemm_kernel(
    A, AS, W, WS, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr,
):
    """A (M,K) int8 with scales (M, K/GROUP); W (N,K) int8 with scales
    (N, K/GROUP); Y (M,N) fp32. Assumes K % GROUP == 0 (callers pad)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, GROUP)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    n_groups = K // GROUP
    for g in range(n_groups):
        kk = g * GROUP + offs_k
        a = tl.load(
            A + offs_m[:, None] * K + kk[None, :],
            mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0)
        w = tl.load(
            W + offs_n[:, None] * K + kk[None, :],
            mask=(offs_n[:, None] < N) & (kk[None, :] < K), other=0)
        dot = tl.dot(a, tl.trans(w), out_dtype=tl.int32)
        sa = tl.load(AS + offs_m * n_groups + g, mask=offs_m < M, other=0.0).to(tl.float32)
        sw = tl.load(WS + offs_n * n_groups + g, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += dot.to(tl.float32) * sa[:, None] * sw[None, :]

    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w8a8_gemm(
    a_codes: torch.Tensor, a_scales: torch.Tensor,
    w_codes: torch.Tensor, w_scales: torch.Tensor,
    group: int = GROUP_SIZE,
) -> torch.Tensor:
    """(M,K)x(N,K)^T -> (M,N) fp32. Inputs as produced by quantize_last_dim
    with the matching `group` (default 128; 32 = the reference parity grid)."""
    M, K = a_codes.shape
    N = w_codes.shape[0]
    if w_codes.shape[1] != K:
        raise ValueError(f"K mismatch: A {K} vs W {w_codes.shape[1]}")
    n_groups = a_scales.shape[-1]
    if w_scales.shape[-1] != n_groups:
        raise ValueError(
            f"scale group mismatch: A {n_groups} vs W {w_scales.shape[-1]}")
    Kp = n_groups * group
    if Kp < K:
        raise ValueError(f"K {K} exceeds the scales' padded length {Kp}")
    if K < Kp:  # partial final group: zero-pad codes (masked loads equivalent)
        a_codes = torch.nn.functional.pad(a_codes, (0, Kp - K))
        w_codes = torch.nn.functional.pad(w_codes, (0, Kp - K))
    K = Kp

    if not a_codes.is_cuda:
        # Reference path: dequantize both operands, fp32 matmul.
        a = a_codes.reshape(M, K // group, group).float() \
            * a_scales.reshape(M, K // group, 1).float()
        w = w_codes.reshape(N, K // group, group).float() \
            * w_scales.reshape(N, K // group, 1).float()
        return a.reshape(M, K) @ w.reshape(N, K).T

    y = torch.empty((M, N), device=a_codes.device, dtype=torch.float32)
    BM, BN = 16, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w8a8_gemm_kernel[grid](
        a_codes, a_scales, w_codes, w_scales, y,
        M, N, K, BM, BN, group, num_warps=4)
    return y
