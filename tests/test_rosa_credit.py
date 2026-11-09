import pytest
import torch

from ksdraft.model.rosa import RosaMixer
from ksdraft.kernels.rosa_credit import (
    engine_available, rosa_credit_forward)

pytestmark = pytest.mark.skipif(
    not engine_available(), reason="credit-engine checkout not present")


def _layer(d=32, s=16, seed=0):
    torch.manual_seed(seed)
    m = RosaMixer(d, s)
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.1)
    return m


def test_engine_forward_matches_torch_reference_exactly():
    """The credit engine and the torch match implement the same hard Rosa:
    longest suffix, latest endpoint, successor read, +-payload."""
    m = _layer().eval()
    x = torch.randn(2, 24, 32)
    q_in, k_in, v_in = m._shifted(x)
    q_full, k_full, v_full = m.q_proj(q_in), m.k_proj(k_in), m.v_proj(v_in)

    # torch reference path
    qbits = (q_full > 0).to(torch.int8)
    kbits = (k_full > 0).to(torch.int8)
    vbits = (v_full > 0).to(torch.int8)
    from ksdraft.model.rosa import _match_payload_sign
    sign = _match_payload_sign(qbits.permute(0, 2, 1), kbits.permute(0, 2, 1),
                               vbits.permute(0, 2, 1))
    ref = (m.payload[None, :, None] * sign).permute(0, 2, 1)

    got = rosa_credit_forward(q_full, k_full, v_full, m.payload,
                              credit_order="first")
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_credit_path_trains_symbol_projections():
    """The point of the engine: q/k/v/time-mix gradients now exist."""
    m = _layer().train()
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = m(x)  # training mode -> credit path
    y.square().mean().backward()
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                 "out_proj.weight", "payload",
                 "time_mix_q", "time_mix_k", "time_mix_v"):
        p = m.get_parameter(name) if "." not in name or "mix" in name \
            else dict(m.named_parameters())[name]
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert torch.isfinite(x.grad).all()


def test_model_end_to_end_with_credit_engine():
    from ksdraft.config import KSDraftConfig
    from ksdraft.model import KSDraftModel
    torch.manual_seed(0)
    cfg = KSDraftConfig(vocab_size=64, layers=2, d_model=48, head_size=16,
                        rosa_streams=8, ks3_charts=24)
    model = KSDraftModel(cfg).train()
    with torch.no_grad():
        model.head.weight.normal_(0, 0.02)
        for b in model.blocks:
            b.rosa.out_proj.weight.normal_(0, 0.02)
            b.ffn.up.weight.normal_(0, 0.02)
    tokens = torch.randint(0, 64, (2, 12))
    logits, feats = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    rosa_q = model.blocks[0].rosa.q_proj.weight
    assert rosa_q.grad is not None and torch.isfinite(rosa_q.grad).all()
