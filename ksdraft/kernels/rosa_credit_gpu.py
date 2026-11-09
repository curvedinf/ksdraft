"""GPU-native Rosa counterfactual credits (exact, batched, chunked).

Computes the same first-order single-bit finite credits as the reference
engine: for every q/k/v bit, flip it, recompute the whole-window match, and
project the output change on the incoming gradient; credits map to
projection space through the Bernoulli factor sigmoid(x)(1-sigmoid(x))
(the engine's credit_to_logit='bernoulli', tau=1).

Exactness: every perturbation is a full re-match (no locality assumption) —
the same quantity as the engine's first-order credit; validated against the
engine's gradients in tests.

Vectorization: streams are independent; each chunk of C streams builds a
(C, 3T, T) perturbation tensor and runs ONE batched match. Cost
O(S * 3T^2 * Lmax) boolean ops — practical for training windows (T <= 63:
the exact integer-code bound); larger windows fall back to the CPU engine.
"""

from __future__ import annotations

import torch

from ..model.rosa import _match_payload_sign

MAX_WINDOW = 63  # exact int64 code width (1-bit streams)


def _chunk_credits(
    sign_base: torch.Tensor, grad: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    """sign_base/grad: (C, T); bits: (C, T) int8 -> credits (3, C, T)."""
    C, T = q.shape
    n = 3 * T
    base = torch.stack((q, k, v), dim=1)                       # (C, 3, T)
    pert = base.unsqueeze(1).expand(C, n, 3, T).clone()        # (C, 3T, 3, T)
    idx = torch.arange(n, device=q.device)
    which = idx // T
    pos = idx % T
    ci = torch.arange(C, device=q.device)[:, None]
    flipped = 1 - pert[ci, idx[None, :], which[None, :], pos[None, :]]
    pert[ci, idx[None, :], which[None, :], pos[None, :]] = flipped

    qp = pert[:, :, 0].reshape(C * n, T)
    kp = pert[:, :, 1].reshape(C * n, T)
    vp = pert[:, :, 2].reshape(C * n, T)
    sign_pert = _match_payload_sign(qp, kp, vp)                # (C*3T, T)

    delta = (sign_pert.reshape(C, n, T) - sign_base.unsqueeze(1))
    credits = (delta * grad.unsqueeze(1)).sum(-1)              # (C, 3T)
    return credits.reshape(C, 3, T).permute(1, 0, 2)           # (3, C, T)


class HardRosaGPU(torch.autograd.Function):
    """q/k/v: (B, T, S) pre-bit projections; payload: (S,) -> hard (B, T, S).

    Forward is the vectorized longest-suffix match (identical semantics to
    the reference path — `_match_payload_sign`); backward is exact batched
    single-bit counterfactual credits with the Bernoulli logit map."""

    @staticmethod
    def forward(ctx, q, k, v, payload):
        if q.shape[1] > MAX_WINDOW:
            raise ValueError(
                f"GPU exact credits support T <= {MAX_WINDOW} (int64 code "
                f"width); got T={q.shape[1]}. Use the CPU engine or chunk "
                "the sequence.")
        qbits = (q > 0).to(torch.int8)
        kbits = (k > 0).to(torch.int8)
        vbits = (v > 0).to(torch.int8)
        sign = _match_payload_sign(
            qbits.permute(0, 2, 1), kbits.permute(0, 2, 1), vbits.permute(0, 2, 1)
        )                                                        # (B, S, T)
        ctx.save_for_backward(q, k, v, sign, payload)
        return (payload[None, :, None] * sign).permute(0, 2, 1)  # (B, T, S)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, sign, payload = ctx.saved_tensors
        B, S, T = sign.shape
        gq = torch.zeros(B, T, S, device=q.device, dtype=q.dtype)
        gk = torch.zeros_like(gq)
        gv = torch.zeros_like(gq)
        grad_hard = grad_out.permute(0, 2, 1).float()            # (B, S, T)
        # y = payload * sign  =>  dL/dsign = grad_y * payload (the match
        # gradient; invisible when payload == 1)
        grad_sign = grad_hard * payload[None, :, None].float()
        qbits = (q > 0).to(torch.int8).permute(0, 2, 1)
        kbits = (k > 0).to(torch.int8).permute(0, 2, 1)
        vbits = (v > 0).to(torch.int8).permute(0, 2, 1)

        chunk = max(1, min(S, (1 << 21) // max(1, 3 * T * T)))   # scratch cap
        for b in range(B):
            for s0 in range(0, S, chunk):
                s1 = min(S, s0 + chunk)
                credits = _chunk_credits(
                    sign[b, s0:s1], grad_sign[b, s0:s1],
                    qbits[b, s0:s1], kbits[b, s0:s1], vbits[b, s0:s1])
                # engine definition: delta = (1 - 2*bit) * C_X, then the
                # bernoulli logit map with tau = 1
                for which, bits in ((0, qbits), (1, kbits), (2, vbits)):
                    delta = credits[which] * (1 - 2 * bits[b, s0:s1].float())
                    out = (gq, gk, gv)[which]
                    out[b, :, s0:s1] = delta.T.to(q.dtype)

        def bernoulli(pre):  # sigmoid(x)(1-sigmoid(x))/tau, tau = 1
            s = torch.sigmoid(pre)
            return (s * (1 - s)).to(pre.dtype)

        gp = (grad_hard * sign.float()).sum(dim=(0, 2))          # d y / d payload
        return (gq * bernoulli(q), gk * bernoulli(k), gv * bernoulli(v),
                gp.to(payload.dtype))
