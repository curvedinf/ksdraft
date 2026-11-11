# SPDX-License-Identifier: Apache-2.0
"""Fused Rosa match core: exact forward, STE/payload backward."""

import pytest
import torch

from ksdraft.kernels.rosa_match import rosa_match
from ksdraft.model.rosa import _match_payload_sign_scan


def _refs(q, k, v, payload):
    with torch.no_grad():
        qb = (q > 0).to(torch.int8).permute(0, 2, 1)
        kb = (k > 0).to(torch.int8).permute(0, 2, 1)
        vb = (v > 0).to(torch.int8).permute(0, 2, 1)
        sign = _match_payload_sign_scan(qb, kb, vb)       # (B, S, T)
    hard = (payload[None, :, None] * sign).permute(0, 2, 1)
    hard = hard + (q - q.detach()) + (k - k.detach()) + (v - v.detach())
    return hard


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_rosa_match_forward_exact():
    torch.manual_seed(0)
    B, T, S = 3, 16, 96
    q = torch.randn(B, T, S, device="cuda")
    k = torch.randn(B, T, S, device="cuda")
    v = torch.randn(B, T, S, device="cuda")
    pay = torch.rand(S, device="cuda") + 0.5
    out = rosa_match(q, k, v, pay)
    ref = _refs(q, k, v, pay)
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_rosa_match_backward_ste_payload():
    torch.manual_seed(1)
    B, T, S = 2, 16, 80
    q = torch.randn(B, T, S, device="cuda", requires_grad=True)
    k = torch.randn(B, T, S, device="cuda", requires_grad=True)
    v = torch.randn(B, T, S, device="cuda", requires_grad=True)
    pay = (torch.rand(S, device="cuda") + 0.5).requires_grad_(True)
    g = torch.randn(B, T, S, device="cuda")

    out = rosa_match(q, k, v, pay)
    out.backward(g)
    gf = [q.grad.clone(), k.grad.clone(), v.grad.clone(), pay.grad.clone()]

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    pay2 = pay.detach().clone().requires_grad_(True)
    ref = _refs(q2, k2, v2, pay2)
    ref.backward(g)
    gr = [q2.grad, k2.grad, v2.grad, pay2.grad]
    for a, b in zip(gf, gr):
        assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_rosa_mixer_module_parity():
    from ksdraft.model.rosa import RosaMixer

    torch.manual_seed(2)
    m = RosaMixer(32, streams=48, bits=1).cuda()
    x = torch.randn(2, 16, 32, device="cuda", requires_grad=True)
    out = m(x)
    out.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters()}

    m2 = RosaMixer(32, streams=48, bits=1).cuda()
    m2.load_state_dict(m.state_dict())
    m2.credit_mode = "ste_fallback"   # force the reference path
    import ksdraft.kernels.rosa_match as rm
    old = rm._HAS_TRITON
    rm._HAS_TRITON = False
    try:
        x2 = x.detach().clone().requires_grad_(True)
        out2 = m2(x2)
        out2.square().mean().backward()
    finally:
        rm._HAS_TRITON = old
    assert torch.allclose(out, out2, atol=1e-5)
    for n, p in m2.named_parameters():
        assert torch.allclose(p.grad, grads[n], atol=1e-5), n
