import pytest
import torch

from ksdraft.kernels.rosa_credit import engine_available, rosa_credit_forward
from ksdraft.kernels.rosa_credit_gpu import HardRosaGPU, MAX_WINDOW
from ksdraft.model.rosa import RosaMixer

pytestmark = pytest.mark.skipif(
    not engine_available(), reason="credit-engine checkout not present")


def _case(seed, B=2, T=16, S=8):
    torch.manual_seed(seed)
    q = torch.randn(B, T, S) * 2
    k = torch.randn(B, T, S) * 2
    v = torch.randn(B, T, S) * 2
    payload = torch.randn(S).abs() + 0.5
    g = torch.randn(B, T, S)
    return q, k, v, payload, g


def test_gpu_credits_match_engine_first_order():
    cos = lambda a, b: torch.nn.functional.cosine_similarity(
        a.detach().flatten().float(), b.detach().flatten().float(), dim=0).item()
    for seed in range(3):
        q, k, v, payload, g = _case(seed)
        q1, k1, v1, p1 = (t.clone().requires_grad_(True) for t in (q, k, v, payload))
        y1 = rosa_credit_forward(q1, k1, v1, p1, credit_order="first")
        y1.backward(g)
        q2, k2, v2, p2 = (t.clone().requires_grad_(True) for t in (q, k, v, payload))
        y2 = HardRosaGPU.apply(q2, k2, v2, p2)
        y2.backward(g)
        assert torch.equal(y1.detach(), y2.detach())          # forward exact
        for a, b in ((q1, q2), (k1, k2), (v1, v2), (p1, p2)):
            assert cos(a.grad, b.grad) > 0.9999, seed


def test_gpu_credits_window_guard():
    q, k, v, payload, g = _case(0, T=MAX_WINDOW + 1, S=2)
    with pytest.raises(ValueError):
        HardRosaGPU.apply(q.requires_grad_(True), k.requires_grad_(True),
                          v.requires_grad_(True), payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_rosa_mixer_gpu_training_path():
    torch.manual_seed(0)
    m = RosaMixer(64, 16).cuda().train()
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.05)
    x = torch.randn(2, 16, 64, device="cuda")
    y = m(x)  # CUDA training -> HardRosaGPU
    y.square().mean().backward()
    for name in ("payload", "q_proj.weight", "k_proj.weight", "v_proj.weight"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
