"""Fused Rosa hard-match core (training path).

The reference computes the longest-suffix match with a dense
(B, S, T, T) equality matrix, a log-T doubling prefix-max, a second full-size
``where``/max for the tie-break, and a gather — ~15 full-size tensor
operations per block. The fused kernel keeps one program per (batch,
stream-block), hoists the K/V bit vectors, and decides each position with
register-resident O(T^2) comparisons. T is a constexpr (training windows are
16 tokens).

Backward is the documented hard-symbol contract: straight-through identity on
the Q/K/V projections plus the exact payload gradient
``sum_{b,t} dhard * sign``. No dense intermediates are stored — only the
sign tensor (int8).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False
    triton = None
    tl = None

BLOCK_S = 64
MAX_T = 32


def fused_enabled(q: torch.Tensor) -> bool:
    if not _HAS_TRITON or not q.is_cuda:
        return False
    if os.environ.get("KSDRAFT_TRITON_ROSA", "1") == "0":
        return False
    return q.shape[1] <= MAX_T and q.dtype in (torch.float16, torch.bfloat16,
                                               torch.float32)


if _HAS_TRITON:
    @triton.jit
    def _match_kernel(q_ptr, k_ptr, v_ptr, pay_ptr, hard_ptr, sign_ptr,
                      S, T: tl.constexpr, BS: tl.constexpr):
        b = tl.program_id(0)
        s0 = tl.program_id(1) * BS
        ss = s0 + tl.arange(0, BS)
        sm = ss < S
        base = b * T * S
        rows = tl.arange(0, T)
        r2 = rows[:, None]
        s2 = ss[None, :]
        koff = base + r2 * S + s2
        kb = tl.load(k_ptr + koff, mask=sm[None, :], other=0.0) > 0.0  # [T,BS]
        vb = tl.load(v_ptr + koff, mask=sm[None, :], other=0.0) > 0.0

        # lcs[t, j] recurrence, held as a [T, BS] state for the previous t.
        prev = tl.zeros([T, BS], dtype=tl.int32)
        idx_shift = tl.maximum(rows - 1, 0)
        shift_idx = tl.broadcast_to(idx_shift[:, None], [T, BS])
        for t in tl.static_range(T):
            qt = tl.load(q_ptr + base + t * S + ss, mask=sm,
                         other=0.0) > 0.0                        # [BS]
            shifted = tl.gather(prev, shift_idx, axis=0)         # prev[j-1]
            shifted = tl.where(r2 > 0, shifted, 0)
            cur = tl.where(qt[None, :] == kb, shifted + 1, 0)    # [T,BS]
            elig = tl.where(r2 < t, cur, 0)
            best = tl.max(elig, axis=0)                          # [BS]
            lat = tl.max(tl.where(elig == best, r2, 0), axis=0)
            succ = lat + 1
            vb_s = tl.sum(tl.where(r2 == succ[None, :], vb.to(tl.int32), 0),
                          axis=0) > 0
            sign = tl.where(best > 0, tl.where(vb_s, 1.0, -1.0), -1.0)
            pay = tl.load(pay_ptr + ss, mask=sm, other=0.0).to(tl.float32)
            tl.store(hard_ptr + base + t * S + ss, pay * sign, mask=sm)
            tl.store(sign_ptr + b * S * T + ss * T + t, sign, mask=sm)
            prev = cur


class _RosaMatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, payload):
        B, T, S = q.shape
        hard = torch.empty((B, T, S), device=q.device, dtype=q.dtype)
        sign = torch.empty((B, S, T), device=q.device, dtype=torch.float32)
        grid = (B, triton.cdiv(S, BLOCK_S))
        _match_kernel[grid](q, k, v, payload, hard, sign, S, T=T,
                            BS=BLOCK_S, num_warps=4)
        ctx.save_for_backward(sign)
        return hard

    @staticmethod
    def backward(ctx, dhard):
        (sign,) = ctx.saved_tensors
        dp = (dhard.float().permute(0, 2, 1) * sign).sum(dim=(0, 2))
        return dhard, dhard, dhard, dp


def rosa_match(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               payload: torch.Tensor) -> torch.Tensor:
    """payload * exact hard-match sign, with the STE/payload backward."""
    return _RosaMatch.apply(q, k, v, payload)
