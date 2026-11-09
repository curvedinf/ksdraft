import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model.rosa import RosaMixer, _match_payload_sign


def _sign(q, k, v):
    """Brute-force reference: longest suffix, latest endpoint, successor bit."""
    T = len(q)
    out = [-1.0] * T
    for t in range(T):
        for L in range(min(t, T - 1), 0, -1):
            pat = q[t - L + 1 : t + 1]
            best = -1
            for j in range(t - 1, L - 2, -1):
                if k[j - L + 1 : j + 1] == pat:
                    best = j
                    break
            if best >= 0:
                out[t] = 1.0 if v[best + 1] else -1.0
                break
    return out


def test_match_payload_sign_hand_case():
    q = [0, 1, 1, 1]
    k = [1, 0, 1, 1]
    v = [1, 0, 0, 1]
    # t=1: L=1 q=1; latest j<1 with k=1 is j=0 -> v[1]=0 -> -1
    # t=2: L=2 q=(1,1): j=1 k=(1,0) no; L=1: j=1 k=0 no, j=0 k=1 yes -> v[1]=0 -> -1
    # t=3: L=3 no; L=2: j=2 k=(0,1) no, j=1 (1,0) no; L=1: j=2 k=1 yes -> v[3]=1 -> +1
    expected = [-1.0, -1.0, -1.0, 1.0]
    got = _match_payload_sign(
        torch.tensor([q]), torch.tensor([k]), torch.tensor([v])
    )[0].tolist()
    assert got == expected


def test_match_payload_sign_vs_bruteforce_random():
    g = torch.Generator().manual_seed(7)
    for _ in range(20):
        T = int(torch.randint(2, 16, (1,), generator=g))
        q = torch.randint(0, 2, (T,), generator=g).tolist()
        k = torch.randint(0, 2, (T,), generator=g).tolist()
        v = torch.randint(0, 2, (T,), generator=g).tolist()
        got = _match_payload_sign(
            torch.tensor([q]), torch.tensor([k]), torch.tensor([v])
        )[0].tolist()
        assert got == _sign(q, k, v), (q, k, v, got, _sign(q, k, v))


def test_match_payload_sign_no_match():
    q = torch.tensor([[1, 1, 1]])
    k = torch.tensor([[0, 0, 0]])
    v = torch.tensor([[1, 1, 1]])
    got = _match_payload_sign(q, k, v)[0].tolist()
    assert got == [-1.0, -1.0, -1.0]


def test_param_count_and_zero_init():
    cfg = KSDraftConfig()
    m = RosaMixer(96, 32)
    n = sum(p.numel() for p in m.parameters())
    # config formula at d=96, s=32: 4*96*32 + 4*96 + 4*32
    assert n == 4 * 96 * 32 + 4 * 96 + 4 * 32

    torch.manual_seed(0)
    m2 = RosaMixer(96, 32).eval()
    x = torch.randn(2, 16, 96)
    assert torch.all(m2(x) == 0)  # zero-init output projection doctrine


def test_forward_shape_and_hard_gradient_flow():
    torch.manual_seed(0)
    m = RosaMixer(64, 16)
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.05)
    x = torch.randn(2, 24, 64)
    y = m(x)
    assert y.shape == (2, 24, 64)

    # eval mode: the torch reference path — payload + output projection
    # receive gradients, symbol projections do not (hard-symbol boundary)
    m.eval()
    y = m(x)
    y.square().mean().backward()
    assert m.payload.grad is not None and m.payload.grad.abs().sum() > 0
    assert m.out_proj.weight.grad is not None and m.out_proj.weight.grad.abs().sum() > 0
    assert m.q_proj.weight.grad is None
    assert m.k_proj.weight.grad is None
    assert m.v_proj.weight.grad is None


def test_training_mode_credit_gradients():
    """With the credit engine present, training mode trains everything."""
    from ksdraft.kernels.rosa_credit import engine_available
    if not engine_available():
        from ksdraft.kernels import rosa_credit  # noqa: F401
        import pytest
        pytest.skip("credit engine not present")
    torch.manual_seed(0)
    m = RosaMixer(64, 16).train()
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.05)
    x = torch.randn(2, 24, 64)
    y = m(x)
    y.square().mean().backward()
    for name in ("payload", "q_proj.weight", "k_proj.weight", "v_proj.weight",
                 "out_proj.weight", "time_mix_q"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
