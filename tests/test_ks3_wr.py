"""Vendored WR KS3 fast path: NL-mode numerics, GS32 grids, gradient flow."""

import os

import pytest
import torch

from ksdraft.model.ks3 import KS3


def _cell(seed=0, d=96, charts=256):
    torch.manual_seed(seed)
    cell = KS3(d, charts).cuda().eval()
    with torch.no_grad():
        cell.coefficients.normal_(0, 0.3)   # exercise the int8 coefficient grid
        cell.down.weight.normal_(0, 0.05)
        cell.up.weight.normal_(0, 0.05)
    return cell


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_wr_forward_matches_reference_loosely():
    cell = _cell()
    x = torch.randn(64, 32, cell.d_model, device="cuda", dtype=torch.float32)
    os.environ["KSDRAFT_KS3_WR"] = "0"
    try:
        with torch.no_grad():
            ref = cell(x)
        os.environ["KSDRAFT_KS3_WR"] = "1"
        with torch.no_grad():
            fast = cell(x)
    finally:
        os.environ.pop("KSDRAFT_KS3_WR", None)
    # GS32 activation grids + the int8 coefficient grid are real quantization;
    # the fast path is NOT bit-parity with the float reference by design.
    assert _rel(fast.float(), ref.float()) < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_wr_nl_ladder():
    cell = _cell(seed=1)
    x = torch.randn(64, 16, cell.d_model, device="cuda", dtype=torch.float32)
    outs = {}
    for nl in (0, 1, 2):
        os.environ["KSDRAFT_KS3_WR_NL"] = str(nl)
        with torch.no_grad():
            outs[nl] = cell(x).float()
    os.environ.pop("KSDRAFT_KS3_WR_NL", None)
    assert _rel(outs[1], outs[0]) < 0.05    # fp16 stages track fp32
    assert _rel(outs[2], outs[0]) < 0.15    # int8 fixed point tracks fp32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_wr_gradients_flow_to_all_surfaces():
    cell = _cell(seed=2).train()
    x = torch.randn(16, 8, cell.d_model, device="cuda", dtype=torch.float32,
                    requires_grad=True)
    y = cell(x)
    loss = (y.float() ** 2).mean()
    loss.backward()
    for name, g in (("coefficients", cell.coefficients.grad),
                    ("raw_scale", cell.raw_scale.grad),
                    ("raw_shift", cell.raw_shift.grad),
                    ("down", cell.down.weight.grad),
                    ("up", cell.up.weight.grad),
                    ("input", x.grad)):
        assert g is not None and torch.isfinite(g).all(), name
    assert cell.coefficients.grad.abs().sum() > 0
    assert cell.raw_scale.grad.abs().sum() > 0
    assert cell.raw_shift.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_wr_gradients_track_reference_direction():
    cell = _cell(seed=3)
    x0 = torch.randn(16, 8, cell.d_model, device="cuda", dtype=torch.float32)
    outs = {}
    grads = {}
    for mode in ("0", "1"):
        os.environ["KSDRAFT_KS3_WR"] = mode
        cell.zero_grad(set_to_none=True)
        xx = x0.clone().requires_grad_(True)
        y = cell(xx)
        loss = (y.float() ** 2).mean()
        loss.backward()
        outs[mode] = loss.item()
        grads[mode] = cell.coefficients.grad.float().flatten()
    os.environ.pop("KSDRAFT_KS3_WR", None)
    cos = torch.nn.functional.cosine_similarity(
        grads["1"], grads["0"], dim=0).item()
    assert cos > 0.9, cos          # quantized path, direction-level agreement
    assert abs(outs["1"] - outs["0"]) < 0.2 * outs["0"]


def test_wr_disabled_on_cpu():
    cell = KS3(96, 64).eval()
    x = torch.randn(2, 8, 96)
    from ksdraft.kernels.ks3_wr import wr_enabled
    assert not wr_enabled(x)
    with torch.no_grad():
        y = cell(x)     # reference path, no triton required
    assert torch.isfinite(y).all()
