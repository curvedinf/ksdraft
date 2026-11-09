"""Parity of the all-INT8 KS3 cell against the actual k-splanifolds reference.

Imports the reference implementation from the local checkout
(~/Projects/k-splanifolds) and compares on the reference's own GS32 grid:

- torch KS3 reference: exact parity (same algorithm, both fp32 geometry)
- INT8 cell fp32 mode: same quantized grid, GEMM accumulation order differs
- INT8 cell int8 mode: ours computes the warp intermediate in FP16 (the
  no-FP32 directive) vs the reference's FP32 — bounded LUT-index divergence
- trainable STE gradients: geometry partials near-exact; projection grads
  looser (our INT8 backward GEMMs vs the reference's FP32 math)
"""

import sys
from pathlib import Path

import pytest
import torch

from ksdraft.model.ks3 import KS3
from ksdraft.kernels.ks3_int8 import KS3Int8Cell

REF_REPO = Path.home() / "Projects/k-splanifolds"
pytestmark = pytest.mark.skipif(
    not REF_REPO.exists(), reason="k-splanifolds reference checkout not present")

sys.path.insert(0, str(REF_REPO))
from experiments.common.ks3 import KS3 as RefKS3          # noqa: E402
from experiments.common.ks3_int8 import KS3Int8 as RefKS3Int8  # noqa: E402


def _make_pair(device="cpu"):
    torch.manual_seed(0)
    ref = RefKS3(96, num_subspaces=368).to(device)
    with torch.no_grad():
        ref.up.weight.normal_(0, 0.05)
    ours = KS3(96, 368).to(device)
    ours.load_state_dict(ref.state_dict())  # identical parameter names
    return ref, ours


def test_torch_reference_exact_parity():
    ref, ours = _make_pair()
    x = torch.randn(8, 96)
    torch.testing.assert_close(ours(x), ref(x), rtol=1e-5, atol=1e-6)


def test_our_ks3_int8_vs_our_ks3_tolerance():
    """Sanity on our own cell: W8A8 + int8 nonlinearities stay within a few %."""
    torch.manual_seed(1)
    cell = KS3(96, 368)
    with torch.no_grad():
        cell.up.weight.normal_(0, 0.05)
    x = torch.randn(8, 96)
    ref_out = cell(x)
    for mode, tol in (("fp32", 0.05), ("fp16", 0.05), ("int8", 0.08)):
        q = KS3Int8Cell(cell, group=128, nonlinear=mode).eval()
        rel = ((q(x) - ref_out).norm() / ref_out.norm()).item()
        assert rel < tol, (mode, rel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU (reference kernels are Triton)")
def test_reference_int8_fp32_mode_parity():
    ref_cell, our_cell = _make_pair("cuda")
    x = torch.randn(64, 96, device="cuda")
    ref_q = RefKS3Int8(ref_cell, nonlinear="fp32").eval()
    our_q = KS3Int8Cell(our_cell, group=32, nonlinear="fp32").eval()
    with torch.no_grad():
        y_ref = ref_q(x)
        y_our = our_q(x)
    torch.testing.assert_close(y_our.float(), y_ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU (reference kernels are Triton)")
def test_reference_int8_int8_mode_parity():
    ref_cell, our_cell = _make_pair("cuda")
    x = torch.randn(64, 96, device="cuda")
    ref_q = RefKS3Int8(ref_cell, nonlinear="int8").eval()
    our_q = KS3Int8Cell(our_cell, group=32, nonlinear="int8").eval()
    with torch.no_grad():
        y_ref = ref_q(x)
        y_our = our_q(x)
    rel = ((y_our.float() - y_ref.float()).norm() / y_ref.float().norm()).item()
    assert rel < 0.05, rel  # FP16 warp intermediate vs FP32: bounded LUT drift


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU (reference kernels are Triton)")
def test_reference_ste_backward_parity():
    ref_cell, our_cell = _make_pair("cuda")
    x = torch.randn(16, 96, device="cuda", requires_grad=True)

    ref_q = RefKS3Int8(ref_cell, nonlinear="int8", trainable=True).train()
    our_q = KS3Int8Cell(our_cell, group=32, nonlinear="int8", trainable=True).train()

    loss_r = ref_q(x).square().mean()
    loss_r.backward()
    loss_o = our_q(x).square().mean()
    loss_o.backward()

    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    # geometry partials: near-exact (same analytic form)
    assert cos(ref_q.master_coefficients.grad, our_q.master_coefficients.grad) > 0.99
    assert cos(ref_q.master_raw_scale.grad, our_q.master_raw_scale.grad) > 0.95
    assert cos(ref_q.master_raw_shift.grad, our_q.master_raw_shift.grad) > 0.95
    # projection grads: INT8 backward GEMMs vs reference FP32 math. ddown's
    # forced short-dim quantization costs the most (~0.82 cosine) — tracked
    # as a float-exception candidate for the acceptance gates.
    assert cos(ref_q.master_down.grad, our_q.master_down.grad) > 0.8
    assert cos(ref_q.master_up.grad, our_q.master_up.grad) > 0.9
    assert torch.isfinite(x.grad).all()
