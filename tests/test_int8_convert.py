import pytest
import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model import KSDraftModel, to_int8, int8_surfaces
from ksdraft.kernels.linear import Int8Linear
from ksdraft.kernels.ks3_int8 import KS3Int8Cell


def _tiny(**kw):
    base = dict(vocab_size=256, layers=2, d_model=96, head_size=32,
                rosa_streams=32, ks3_charts=64)
    base.update(kw)
    return KSDraftConfig(**base)


def _woken_model(cfg, teacher_dim=0):
    torch.manual_seed(0)
    model = KSDraftModel(cfg, teacher_dim=teacher_dim).eval()
    with torch.no_grad():
        model.head.weight.normal_(0, 0.02)
        for b in model.blocks:
            b.rosa.out_proj.weight.normal_(0, 0.05)
            b.ffn.up.weight.normal_(0, 0.05)
    return model


def test_to_int8_swaps_surfaces_and_keeps_budget():
    cfg = _tiny()
    model = _woken_model(cfg, teacher_dim=3 * 32)
    converted = to_int8(model)
    for block in converted.blocks:
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert isinstance(getattr(block.rosa, name), Int8Linear)
        assert isinstance(block.ffn, KS3Int8Cell)
        # float exceptions per doctrine
        assert isinstance(block.ln_attn, torch.nn.LayerNorm)
        assert block.rosa.payload.dtype.is_floating_point
    assert isinstance(converted.teacher_proj, Int8Linear)
    assert int8_surfaces(converted)["swapped_surfaces"] == 2 * 5 + 1


def test_to_int8_forward_within_tolerance():
    cfg = _tiny()
    model = _woken_model(cfg)
    tokens = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        y_ref, feats_ref = model(tokens)
        converted = to_int8(model)
        y_q, _ = converted(tokens)
    rel = ((y_q - y_ref).norm() / y_ref.norm()).item()
    assert rel < 0.15, rel  # int8 Rosa projections + int8 KS3 end to end


def test_to_int8_training_grads_flow():
    cfg = _tiny()
    model = _woken_model(cfg, teacher_dim=3 * 32).train()
    converted = to_int8(model)
    tokens = torch.randint(0, 256, (2, 8))
    ts = torch.randn(2, 8, 96)
    logits, _ = converted(tokens, teacher_states=ts)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    grads = [p.grad for p in converted.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    rosa_q = converted.blocks[0].rosa.q_proj.weight
    assert rosa_q.grad is not None and torch.isfinite(rosa_q.grad).all()


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_to_int8_gpu_and_cpu_paths(device):
    cfg = _tiny()
    model = _woken_model(cfg).to(device)
    tokens = torch.randint(0, 256, (2, 8), device=device)
    with torch.no_grad():
        y_ref, _ = model(tokens)
        y_q, _ = to_int8(model)(tokens)
    rel = ((y_q - y_ref).norm() / y_ref.norm()).item()
    assert rel < 0.15, (device, rel)
