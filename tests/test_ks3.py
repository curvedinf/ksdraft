import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model.ks3 import KS3, spine_horner, spine_basis


def _random_spine_inputs(charts=64, lead=(4,), seed=0):
    g = torch.Generator().manual_seed(seed)
    t = 4.0 * torch.tanh(torch.randn(*lead, charts, 1, generator=g))
    z = torch.tanh(torch.randn(*lead, charts, 1, generator=g))
    coeffs = torch.randn(charts, 1, 6, generator=g)
    return t, z, coeffs


def test_horner_matches_explicit_basis():
    """The Horner path must agree with the explicit Hermite/Chebyshev oracle."""
    for seed in range(5):
        t, z, coeffs = _random_spine_inputs(seed=seed)
        sh, wh = spine_horner(t, z, coeffs)
        sb, wb = spine_basis(t, z, coeffs)
        # t in +-4 makes spine magnitudes ~400; fp32 rounding gives ~1e-5 rel
        torch.testing.assert_close(sh, sb, rtol=2e-5, atol=1e-4)
        torch.testing.assert_close(wh, wb, rtol=2e-5, atol=1e-4)


def test_horner_is_the_module_path():
    """Module forward == Horner evaluation composed with the projections."""
    torch.manual_seed(1)
    m = KS3(32, 64).double().eval()
    x = torch.randn(4, 32, dtype=torch.float64)
    z_coord = m.down(x).clamp(-65504.0, 65504.0).reshape(4, 64, 1)
    t = 4.0 * torch.tanh(z_coord / 4.0)
    zw = torch.tanh(m.warp_scale()[:, None] * (t + m.warp_shift()[:, None]))
    coeffs = m.coefficients.reshape(64, 1, 6)
    spine, warped = spine_horner(t, zw, coeffs)
    expected = m.up(torch.relu(z_coord + spine + warped).square().flatten(-2))
    torch.testing.assert_close(m(x), expected, rtol=1e-10, atol=1e-12)


def test_param_count_matches_config_formula():
    cfg = KSDraftConfig()
    m = KS3(cfg.d_model, cfg.charts)
    n = sum(p.numel() for p in m.parameters())
    assert n == cfg.ks3_params() == 24_673_448


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
        with torch.no_grad():
            m.up.weight.normal_(0, 0.02)
        x = torch.randn(2, 16, 64, dtype=dtype, requires_grad=True)
        y = m(x)
        assert y.shape == (2, 16, 64)
        y.square().mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert x.grad is not None


def test_geometry_runs_fp32_under_bf16():
    """Canonical dtype discipline: bf16 in -> fp32 geometry -> close to fp32 model."""
    torch.manual_seed(2)
    base = KS3(64, 128)
    m32 = KS3(64, 128).eval()
    m32.load_state_dict(base.state_dict())
    mbf = KS3(64, 128).to(torch.bfloat16).eval()
    mbf.load_state_dict(base.state_dict())
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        y32 = m32(x)
        ybf = mbf(x.to(torch.bfloat16)).float()
    torch.testing.assert_close(ybf, y32, rtol=5e-2, atol=5e-2)


def test_coordinate_clamp_keeps_output_finite():
    torch.manual_seed(3)
    m = KS3(32, 64).eval()
    with torch.no_grad():
        m.up.weight.normal_(0, 0.05)
    x = torch.randn(2, 8, 32) * 1e6
    y = m(x)
    assert torch.isfinite(y).all()
