import pytest
import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model import AdjacentInjection, DraftBlock, KSDraftModel


def _tiny(**kw):
    base = dict(vocab_size=256, layers=2, d_model=96, head_size=32,
                rosa_streams=32, ks3_charts=64)
    base.update(kw)
    return KSDraftConfig(**base)


def test_fresh_block_is_identity():
    torch.manual_seed(0)
    from ksdraft.model.spine import SpineCore

    cfg = _tiny()
    core = SpineCore(cfg.d_model)
    block = DraftBlock(cfg, 0, core, cfg.d_model).eval()
    x = torch.randn(2, 16, cfg.d_model)
    y, feat = block(x, None, 0.0)
    assert torch.allclose(y, x, atol=1e-6)  # every additive update zero-init
    assert feat.shape == x.shape


def test_injection_shapes_and_position_shift():
    from ksdraft.model.spine import SpineCore

    core = SpineCore(8)
    inj = AdjacentInjection(64, 8, core)
    u = torch.randn(2, 10, 64)
    h, e = inj(u)
    assert h.shape == u.shape
    assert e.shape == (2, 10, 8)
    # G_p / W zero-init -> the injection is the identity at init
    assert torch.allclose(h, u)


def test_model_forward_backward():
    torch.manual_seed(0)
    cfg = _tiny()
    model = KSDraftModel(cfg)
    # wake the zero-init surfaces so every parameter receives gradient signal
    with torch.no_grad():
        model.head.weight.normal_(0, 0.02)
        for b in model.blocks:
            b.rosa.out_proj.weight.normal_(0, 0.02)
            b.ffn.up.weight.normal_(0, 0.02)
            b.injection.W.weight.normal_(0, 0.02)
        model.spine_core.Gp.weight.normal_(0, 0.02)
        model.spine_core.A.weight.normal_(0, 0.02)
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    logits, features = model(tokens)
    assert logits.shape == (2, 24, cfg.vocab_size)
    assert len(features) == cfg.layers
    assert all(f.shape == (2, 24, cfg.d_model) for f in features)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_tied_head():
    model = KSDraftModel(_tiny())
    assert model.head.weight is model.embedding.weight


def test_teacher_conditioning_seam():
    torch.manual_seed(0)
    cfg = _tiny()
    model = KSDraftModel(cfg, teacher_dim=48)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    ts = torch.randn(2, 8, 48)
    with torch.no_grad():
        base, _ = model(tokens)          # zero-init projection: no effect
        cond, _ = model(tokens, ts)
    assert torch.allclose(base, cond, atol=1e-6)
    with torch.no_grad():
        model.teacher_proj.weight.normal_(0, 0.1)
        cond2, _ = model(tokens, ts)
    assert not torch.allclose(base, cond2, atol=1e-4)
    with pytest.raises(ValueError):
        KSDraftModel(cfg)(tokens, ts)  # seam disabled -> explicit error


def test_bf16_forward():
    torch.manual_seed(0)
    cfg = _tiny()
    model = KSDraftModel(cfg).to(torch.bfloat16)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(tokens)
    assert logits.dtype == torch.bfloat16


def test_full_shape_param_budget():
    """Decided shape on the meta device: core 964,153,248 + injection/LN extras
    + block-head offsets + the DFlash2-style intra-block conv."""
    cfg = KSDraftConfig()
    d0 = cfg.d_model
    rank = 256
    model = KSDraftModel(cfg, device="meta", skip_init=True,
                         transfer_space={
                             "C": torch.empty(cfg.vocab_size, rank,
                                              device="meta"),
                             "rank": rank})
    counts = model.param_counts()
    assert counts["core"] == cfg.total_params() == 964_153_248
    d = cfg.d_model
    ln_params = cfg.layers * 2 * 2 * d + 2 * 2 * d  # 2 norms/block + in/out
    block_head = cfg.block_len * d
    conv = 2 * d + 2 * d * d          # base taps + dynamic projection
    selector = 3 * d * 256            # predecessor/successor/context projections
    # DSpine injection: per-layer R + W (rank 256), shared A/Gq/Gp/b
    spine_layer = 2 * 256 * d
    spine_core = 3 * 256 * 256 + 256
    spine_head = 2 * 256 * d          # head last-write R + W
    assert counts["extras"] == (cfg.layers * spine_layer + spine_core
                                + spine_head + ln_params + block_head + conv
                                + selector)
    assert counts["total"] == counts["core"] + counts["extras"]
