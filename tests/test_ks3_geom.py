# SPDX-License-Identifier: Apache-2.0
"""Fused KS3 geometry: forward/backward parity with the reference chain."""

import pytest
import torch
import torch.nn.functional as F

from ksdraft.kernels.ks3_geom import ks3_geometry
from ksdraft.model.ks3 import spine_horner


def _ref(h, coeffs, raw_scale, raw_shift):
    hf = h.float().clamp(-65504.0, 65504.0)
    t = 4.0 * torch.tanh(hf / 4.0)
    scale = F.softplus(raw_scale.float()) + 1e-4
    shift = 0.5 * torch.tanh(raw_shift.float())
    z = torch.tanh(scale * (t + shift))
    spine, warped = spine_horner(t, z, coeffs.float())
    return F.relu(hf + spine + warped).square()


def _make(dev, dtype, P=48, C=40):
    torch.manual_seed(0)
    h = (torch.randn(P, C, device=dev) * 3).to(dtype).requires_grad_(True)
    coeffs = (torch.randn(C, 6, device=dev) * 0.05).requires_grad_(True)
    rs = torch.randn(C, device=dev).requires_grad_(True)
    rh = (torch.randn(C, device=dev) * 0.1).requires_grad_(True)
    return h, coeffs, rs, rh


def _close(a, b, rel):
    total = a.abs().max().clamp_min(1e-6)
    return (a - b).abs().max() <= rel * total + 1e-6


def _check(dev, dtype, rel):
    h, c, rs, rh = _make(dev, dtype)
    ref = _ref(h, c, rs, rh)
    g = torch.randn_like(ref)
    ref.backward(g)
    gr = [h.grad.clone(), c.grad.clone(), rs.grad.clone(), rh.grad.clone()]

    h2, c2, rs2, rh2 = _make(dev, dtype)
    out = ks3_geometry(h2, c2, rs2, rh2)
    assert out.dtype == h2.dtype
    assert _close(ref.float(), out.float(), rel)
    out.backward(g)
    names = ["h", "coeffs", "raw_scale", "raw_shift"]
    for n, a, b in zip(names, gr, [h2.grad, c2.grad, rs2.grad, rh2.grad]):
        assert _close(a, b, rel), n


def test_ks3_geom_cpu_fp32():
    _check("cpu", torch.float32, 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_ks3_geom_gpu_fp16():
    _check("cuda", torch.float16, 2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_ks3_geom_gpu_fp32():
    _check("cuda", torch.float32, 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_ks3_module_matches_reference_path():
    """The module's fused path matches the pre-fusion module output/grads."""
    from ksdraft.model.ks3 import KS3

    torch.manual_seed(0)
    m = KS3(32, num_charts=48).cuda()
    x = torch.randn(2, 16, 32, device="cuda", requires_grad=True)
    out = m(x)
    out.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters()}

    m2 = KS3(32, num_charts=48).cuda()
    m2.load_state_dict(m.state_dict())
    x2 = x.detach().clone().requires_grad_(True)
    out2 = m2(x2)
    out2.square().mean().backward()
    assert torch.allclose(out.float(), out2.float(), atol=2e-2)
    for n, p in m2.named_parameters():
        assert torch.allclose(p.grad, grads[n], atol=2e-2), n
