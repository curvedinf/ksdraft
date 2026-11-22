# SPDX-License-Identifier: Apache-2.0
"""Chunked head losses: numerical parity with the reference path."""

import pytest
import torch
import torch.nn.functional as F

from ksdraft.kernels.fused_train import (block_position_weights, linear_ce,
                                         linear_soft_ce, wtv_from_logp)
from ksdraft.kernels.quant import dequantize_last_dim, quantize_last_dim


def _dq(x):
    """Quantize-dequantize operand (identical codes/scales as the int8 path)."""
    codes, scales = quantize_last_dim(x.detach().float())
    return dequantize_last_dim(codes, scales)


def _ste(x):
    """STE identity so the int8 reference has autograd w.r.t. the masters."""
    return _dq(x) + (x - x.detach())


def _ref_hard(x, w, t, pw):
    lg = (_ste(x) @ _ste(w).t()).float()
    ce = F.cross_entropy(lg, t, reduction="none")
    return (pw * ce).sum() / pw.sum()


def test_hard_ce_parity_and_grads():
    torch.manual_seed(0)
    V, D, N = 97, 16, 21
    x = torch.randn(N, D, requires_grad=True)
    w = torch.randn(V, D, requires_grad=True)
    t = torch.randint(0, V, (N,))
    pw = torch.rand(N) + 0.1
    ref = _ref_hard(x, w, t, pw)
    ref.backward()
    gx, gw = x.grad.clone(), w.grad.clone()
    x2, w2 = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
    out, per = linear_ce(x2, w2, t, pw, chunk=5)
    out.backward()
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(x2.grad, gx, atol=1e-6)
    assert torch.allclose(w2.grad, gw, atol=1e-6)
    assert per.shape == (N,)


def test_hard_ce_unit_weights():
    torch.manual_seed(1)
    V, D, N = 41, 8, 12
    x = torch.randn(N, D)
    w = torch.randn(V, D)
    t = torch.randint(0, V, (N,))
    ref = F.cross_entropy((_dq(x) @ _dq(w).t()).float(), t)
    out, _ = linear_ce(x, w, t, None, chunk=4)
    assert torch.allclose(out, ref, atol=1e-5)
    # the int8 loss stays within 2% of the fp32 loss (quantization sanity)
    fp32 = F.cross_entropy(x @ w.t(), t)
    assert (out - fp32).abs() / fp32.abs() < 0.02


def test_soft_ce_kl_and_wtv_grads():
    torch.manual_seed(2)
    V, D, N, K = 53, 12, 9, 5
    x = torch.randn(N, D, requires_grad=True)
    w = torch.randn(V, D, requires_grad=True)
    ids = torch.randint(0, V, (N, K))
    pt = torch.rand(N, K)
    pt = pt / pt.sum(-1, keepdim=True)

    lg = (_ste(x) @ _ste(w).t()).float()
    logp = F.log_softmax(lg, -1)
    on = logp.gather(-1, ids)
    ref = -(pt * on).sum(-1).mean() + 0.3 * wtv_from_logp(on, pt)
    ref.backward()
    gx, gw = x.grad.clone(), w.grad.clone()

    x2, w2 = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
    kl, on2 = linear_soft_ce(x2, w2, ids, pt, None, chunk=4)
    out = kl + 0.3 * wtv_from_logp(on2, pt)
    out.backward()
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(x2.grad, gx, atol=1e-6)
    assert torch.allclose(w2.grad, gw, atol=1e-6)
    assert torch.allclose(on2, on, atol=1e-6)


def test_position_weights():
    w = block_position_weights(4, 0.8, "cpu")
    assert torch.allclose(w, torch.tensor([1.0, 0.8, 0.64, 0.512]))
    assert torch.equal(block_position_weights(4, 1.0, "cpu"), torch.ones(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_hard_ce_fp16_gpu():
    torch.manual_seed(3)
    V, D, N = 1021, 64, 33
    x = torch.randn(N, D, device="cuda", dtype=torch.float16)
    w = torch.randn(V, D, device="cuda", dtype=torch.float16)
    t = torch.randint(0, V, (N,), device="cuda")
    ref = F.cross_entropy((_dq(x) @ _dq(w).t()).float(), t)
    out, _ = linear_ce(x, w, t, None, chunk=7)
    assert torch.allclose(out, ref, atol=2e-3)
