# SPDX-License-Identifier: Apache-2.0
"""BlockConv: identity at init, causal, gradient-carrying."""

import torch

from ksdraft.model.blockconv import BlockConv


def test_identity_at_init():
    m = BlockConv(32)
    x = torch.randn(3, 5, 32)
    assert torch.equal(m(x), x)


def test_causal_and_lookback():
    m = BlockConv(32)
    with torch.no_grad():
        m.base[1].fill_(1.0)
        m.base[0].zero_()
    x = torch.randn(2, 6, 32)
    y = m(x)
    # out_k = x_{k-1}: position 0 is zero, later positions shift
    assert torch.equal(y[:, 0], torch.zeros_like(y[:, 0]))
    assert torch.equal(y[:, 1:], x[:, :-1])
    # changing a later position cannot affect earlier outputs
    x2 = x.clone()
    x2[:, 4] += 1.0
    y2 = m(x2)
    assert torch.equal(y[:, :4], y2[:, :4])


def test_gradients_flow():
    m = BlockConv(32)
    x = torch.randn(2, 4, 32, requires_grad=True)
    m(x).square().mean().backward()
    assert x.grad is not None
    assert m.dynamic.weight.grad is not None
    assert m.base.grad is not None


def test_group_shared_dynamic():
    m = BlockConv(32, group_size=16)
    m.dynamic.weight.data.normal_()
    x = torch.randn(1, 4, 32)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
