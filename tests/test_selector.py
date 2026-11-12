# SPDX-License-Identifier: Apache-2.0
"""DFlash2 pairwise selector: init equivalence, score math, walk, loss."""

import torch

from ksdraft.model.selector import BlockSelector


def test_zero_init_is_unary():
    torch.manual_seed(0)
    sel = BlockSelector(16, rank=8)
    n, k = 5, 4
    h = torch.randn(n, 16)
    unary = torch.randn(n, k)
    cand = torch.randn(n, k, 16)
    prev = torch.randn(n, 16)
    assert torch.equal(sel.score_candidates(h, unary, cand, prev), unary)


def test_score_math():
    torch.manual_seed(1)
    sel = BlockSelector(6, rank=3)
    with torch.no_grad():
        sel.a_proj.weight.normal_()
        sel.b_proj.weight.normal_()
        sel.ctx_proj.weight.normal_()
    n, k, d, r = 2, 4, 6, 3
    h = torch.randn(n, d); unary = torch.randn(n, k)
    cand = torch.randn(n, k, d); prev = torch.randn(n, d)
    out = sel.score_candidates(h, unary, cand, prev)
    a = prev @ sel.a_proj.weight.t()
    g = h @ sel.ctx_proj.weight.t()
    b = cand @ sel.b_proj.weight.t()
    manual = unary + ((a * g)[:, None, :] * b).sum(-1)
    assert torch.allclose(out, manual, atol=1e-6)


def test_selector_loss_finite_and_grads():
    torch.manual_seed(2)
    from ksdraft.config import KSDraftConfig
    from ksdraft.model import KSDraftModel
    cfg = KSDraftConfig(vocab_size=128, d_model=32, layers=2, head_size=16,
                        rosa_streams=16, ks3_charts=64, block_len=8)
    m = KSDraftModel(cfg)
    b, n, d = 3, 4, 32
    hs = torch.randn(b, n, d, requires_grad=True)
    tokens = torch.randint(0, cfg.vocab_size, (b, 20))
    loss = m.selector_loss(hs, tokens, 15, n, topk=8)
    assert torch.isfinite(loss)
    loss.backward()
    assert hs.grad is not None
    assert m.block_selector.a_proj.weight.grad is not None


def test_walk_returns_candidates():
    torch.manual_seed(3)
    from ksdraft.config import KSDraftConfig
    from ksdraft.model import KSDraftModel
    cfg = KSDraftConfig(vocab_size=128, d_model=32, layers=2, head_size=16,
                        rosa_streams=16, ks3_charts=64, block_len=8)
    m = KSDraftModel(cfg).eval()
    ctx = torch.randint(0, cfg.vocab_size, (2, 16))
    toks = m.propose_block_tokens(ctx, 4, topk=8)
    assert toks.shape == (2, 4)
    assert toks.min() >= 0 and toks.max() < cfg.vocab_size


def test_lwr_identity_at_init_matches_unary_walk():
    torch.manual_seed(4)
    from ksdraft.config import KSDraftConfig
    from ksdraft.model import KSDraftModel
    cfg = KSDraftConfig(vocab_size=128, d_model=32, layers=2, head_size=16,
                        rosa_streams=16, ks3_charts=64, block_len=8)
    m = KSDraftModel(cfg).eval()
    ctx = torch.randint(0, cfg.vocab_size, (2, 16))
    # head_write.W / G_p are zero at init -> refinement is the identity and
    # the walk must equal the unary argmax
    blog = m.block_logits(ctx, 4).float()
    walked = m.propose_block_tokens(ctx, 4, topk=8)
    assert torch.equal(walked, blog.argmax(-1))


def test_lwr_loss_finite_and_grads():
    torch.manual_seed(5)
    from ksdraft.config import KSDraftConfig
    from ksdraft.model import KSDraftModel
    cfg = KSDraftConfig(vocab_size=128, d_model=32, layers=2, head_size=16,
                        rosa_streams=16, ks3_charts=64, block_len=8)
    m = KSDraftModel(cfg)
    with torch.no_grad():
        m.head_write.W.weight.normal_(0, 0.05)
        m.spine_core.Gp.weight.normal_(0, 0.05)
    b, n, d = 3, 4, 32
    hs = torch.randn(b, n, d, requires_grad=True)
    tokens = torch.randint(0, cfg.vocab_size, (b, 20))
    loss = m.lwr_loss(hs, tokens, 15, n, topk=8)
    assert torch.isfinite(loss)
    loss.backward()
    assert hs.grad is not None
    assert m.head_write.W.weight.grad is not None
    assert m.head_write.R.weight.grad is not None


def test_lwr_walk_returns_candidates():
    torch.manual_seed(6)
    from ksdraft.config import KSDraftConfig
    from ksdraft.model import KSDraftModel
    cfg = KSDraftConfig(vocab_size=128, d_model=32, layers=2, head_size=16,
                        rosa_streams=16, ks3_charts=64, block_len=8)
    m = KSDraftModel(cfg).eval()
    with torch.no_grad():
        m.head_write.W.weight.normal_(0, 0.05)
    ctx = torch.randint(0, cfg.vocab_size, (2, 16))
    tok = m.propose_block_tokens(ctx, 4, topk=8)
    assert tok.shape == (2, 4)
    assert tok.min() >= 0 and tok.max() < cfg.vocab_size
