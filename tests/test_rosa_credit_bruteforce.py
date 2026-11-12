# SPDX-License-Identifier: Apache-2.0
"""Exact GPU counterfactual credits vs brute-force single-bit flips.

The upstream engine checkout is not vendored, so this test validates
HardRosaGPU against the definition directly: flip one Q/K/V bit, recompute
the hard match, project the output change on the incoming gradient, apply
the Bernoulli logit map.
"""

import pytest
import torch

from ksdraft.kernels.rosa_credit_gpu import HardRosaGPU
from ksdraft.model.rosa import _match_payload_sign

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs GPU")


def _sign(qb, kb, vb):
    return _match_payload_sign(qb.permute(0, 2, 1), kb.permute(0, 2, 1),
                               vb.permute(0, 2, 1))


def test_gpu_credits_match_bruteforce():
    torch.manual_seed(0)
    B, T, S = 1, 6, 2
    q, k, v = (torch.randn(B, T, S, device="cuda") * 2 for _ in range(3))
    pay = torch.rand(S, device="cuda") + 0.5
    g = torch.randn(B, T, S, device="cuda")

    q1, k1, v1, p1 = (t.clone().requires_grad_(True) for t in (q, k, v, pay))
    y = HardRosaGPU.apply(q1, k1, v1, p1)
    y.backward(g)

    with torch.no_grad():
        qb = (q > 0).to(torch.int8)
        kb = (k > 0).to(torch.int8)
        vb = (v > 0).to(torch.int8)
        sign = _sign(qb, kb, vb)
        grad_sign = g.permute(0, 2, 1) * pay[None, :, None]

        def credit(bits, b, t, s):
            b2 = bits.clone()
            b2[b, t, s] = 1 - b2[b, t, s]
            if bits is qb:
                s2 = _sign(b2, kb, vb)
            elif bits is kb:
                s2 = _sign(qb, b2, vb)
            else:
                s2 = _sign(qb, kb, b2)
            return ((s2 - sign) * grad_sign).sum()

        def bernoulli(pre):
            p = torch.sigmoid(pre)
            return p * (1 - p)

        for bits, pre, grad in ((qb, q, q1.grad), (kb, k, k1.grad),
                                (vb, v, v1.grad)):
            for b in range(B):
                for t in range(T):
                    for s in range(S):
                        c = credit(bits, b, t, s)
                        expect = ((1 - 2 * int(bits[b, t, s])) * c
                                  * bernoulli(pre[b, t, s]))
                        assert torch.allclose(grad[b, t, s], expect,
                                              atol=1e-5), (b, t, s)
