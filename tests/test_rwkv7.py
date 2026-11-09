import math

import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model.rwkv7 import RWKV7TimeMix, delta_rule_recurrence


def test_delta_rule_recurrence_shapes_and_bounds():
    g = torch.Generator().manual_seed(0)
    B, T, H, W = 2, 8, 3, 16
    read = torch.randn(B, T, H, W, generator=g)
    decay = torch.rand(B, T, H, 1, generator=g) * 0.9 + 0.05  # in (0.05, 0.95)
    key = torch.randn(B, T, H, W, generator=g)
    value = torch.randn(B, T, H, W, generator=g)
    nk = torch.nn.functional.normalize(key, dim=-1)
    rate = torch.rand(B, T, H, 1, generator=g)
    out = delta_rule_recurrence(read, decay, key, value, nk, rate)
    assert out.shape == (B, T, H, W)
    assert torch.isfinite(out).all()


def test_timemix_param_count():
    cfg = KSDraftConfig()
    m = RWKV7TimeMix(cfg.d_model, layer=5, layers=cfg.layers, head_size=cfg.head_size)
    n = sum(p.numel() for p in m.parameters())
    assert n == cfg.mixer_params() == 4 * 1536 * 1536 + 334 * 1536


def test_timemix_zero_init_output_and_causality():
    torch.manual_seed(0)
    m = RWKV7TimeMix(128, layer=1, layers=4, head_size=64).eval()
    x = torch.randn(2, 12, 128)
    out, fv = m(x, None)
    assert torch.all(out == 0)  # zero-init output projection

    with torch.no_grad():
        m.output.weight.normal_(0, 0.05)
        x2 = x.clone()
        x2[0, 7] += 1.0
        o1, _ = m(x, None)
        o2, _ = m(x2, None)
    assert torch.allclose(o1[0, :7], o2[0, :7], atol=1e-5)  # causal
    assert not torch.allclose(o1[0, 7:], o2[0, 7:], atol=1e-4)


def test_timemix_first_value_carried():
    torch.manual_seed(1)
    m = RWKV7TimeMix(128, layer=1, layers=4, head_size=64).eval()
    with torch.no_grad():
        m.output.weight.normal_(0, 0.05)
    x = torch.randn(1, 10, 128)
    fv_seed = torch.randn_like(x)
    _, fv_out = m(x, fv_seed)
    assert torch.equal(fv_out, fv_seed)  # first block sets it; later blocks carry
