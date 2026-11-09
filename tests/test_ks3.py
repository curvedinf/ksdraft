import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model.ks3 import KS3


def test_param_count_matches_config_formula():
    cfg = KSDraftConfig()
    m = KS3(cfg.d_model, cfg.charts)
    n = sum(p.numel() for p in m.parameters())
    assert n == cfg.ks3_params() == 18_135_040


def test_zero_output_projection_doctrine():
    torch.manual_seed(0)
    m = KS3(96, 368).eval()
    x = torch.randn(2, 32, 96)
    y = m(x)
    assert torch.all(y == 0)  # fresh KS3 contributes nothing


def test_forward_shape_and_grads():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        m = KS3(64, 128).to(dtype)
        # perturb the zero-init so gradients reach every parameter
        with torch.no_grad():
            m.up.weight.normal_(0, 0.02)
            m.coefficients.normal_(0, 0.1)
        x = torch.randn(2, 16, 64, dtype=dtype, requires_grad=True)
        y = m(x)
        assert y.shape == (2, 16, 64)
        y.square().mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert x.grad is not None


def test_geometry_matches_kernel_formulas():
    """The spine geometry must match the fused-kernel fp32 branch exactly."""
    torch.manual_seed(1)
    m = KS3(32, 64)
    with torch.no_grad():
        m.coefficients.normal_(0, 0.5)
        m.raw_scale.normal_(0, 0.5)
        m.raw_shift.normal_(0, 0.5)
    x = torch.randn(4, 32)
    z = m.down(x)
    t = 4 * torch.tanh(z * 0.25)
    ws, wsh = m.warp_scale(), m.warp_shift()
    zw = torch.tanh(ws * (t + wsh))
    c0, c1, c2, c3, c4, c5 = m.coefficients.unbind(-1)
    a = 2 * c0 - 2 * c1 + c2 + c3
    b = -3 * c0 + 3 * c1 - 2 * c2 - c3
    poly = ((a * t + b) * t + c2) * t + c0
    warp = ((4 * c5 * zw + 2 * c4) * zw - 3 * c5) * zw - c4
    h = torch.relu(z + poly + warp).square()
    expected = m.up(h)
    assert torch.allclose(m(x), expected, atol=1e-6)
