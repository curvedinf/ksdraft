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
    cfg = _tiny()
    block = DraftBlock(cfg, layer=0).eval()
    x = torch.randn(2, 16, cfg.d_model)
    y, fv, feat = block(x, None)
    assert torch.allclose(y, x, atol=1e-6)  # every additive update zero-init


def test_injection_shapes_and_position_shift():
    inj = AdjacentInjection(64)
    z = torch.randn(2, 10, 64)
    z2, p = inj(z)
    assert z2.shape == p.shape == z.shape
    # feature head zero-init -> injection contributes nothing, z2 == z
    assert torch.allclose(z2, z)


def test_model_forward_backward():
    torch.manual_seed(0)
    cfg = _tiny()
    model = KSDraftModel(cfg)
    # wake the zero-init surfaces so every parameter receives gradient signal
    with torch.no_grad():
        model.head.weight.normal_(0, 0.02)
        for b in model.blocks:
            b.time_mix.output.weight.normal_(0, 0.02)
            b.rosa.out_proj.weight.normal_(0, 0.02)
            b.ffn.up.weight.normal_(0, 0.02)
            b.injection.inject.up.weight.normal_(0, 0.02)
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
    """Decided shape on the meta device: core 980,361,216 + injection/LN extras."""
    cfg = KSDraftConfig()
    model = KSDraftModel(cfg, device="meta", skip_init=True)
    counts = model.param_counts()
    assert counts["core"] == cfg.total_params() == 980_361_216
    d = cfg.d_model
    injection_per_layer = (2 * 64 + 6 * 32) * d + d
    ln_params = cfg.layers * 3 * 2 * d + 2 * 2 * d  # 3 norms/block + in/out
    assert counts["extras"] == cfg.layers * injection_per_layer + ln_params
    assert counts["total"] == 992_421_888  # ~0.99B, inside the decided ~1B
