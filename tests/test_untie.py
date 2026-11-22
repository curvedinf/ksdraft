"""Untied draft (DFly2 pattern): two borrowed tables, W4 unembed dequant."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ksdraft.config import KSDraftConfig
from ksdraft.export import export_model, load_export
from ksdraft.model import KSDraftModel
from ksdraft.teacher import KVFixture
from ksdraft.train import Trainer, TrainConfig, WindowSampler


def _cfg(**kw):
    base = dict(vocab_size=128, layers=2, d_model=96, head_size=32,
                rosa_streams=32, ks3_charts=64, kv_dim=64, n_kv_heads=4,
                kv_window=8, dtype="fp32")
    base.update(kw)
    return KSDraftConfig(**base)


def _kv(model, tokens):
    fx = KVFixture(model.config.layers, model.config.kv_dim,
                   kv_window=model.config.kv_window, seed=3)
    sig = fx.signals(tokens.cpu())
    return tuple(t.to(tokens.device, dtype=torch.float32) for t in sig.kv)


def _woken(cfg, seed=0):
    torch.manual_seed(seed)
    model = KSDraftModel(cfg).eval()
    with torch.no_grad():
        model.embedding.weight.normal_(0, 0.02)
        if model.head.weight is not model.embedding.weight:
            model.head.weight.normal_(0, 0.02)
        for b in model.blocks:
            if hasattr(b, "attn"):
                b.attn.out_proj.weight.normal_(0, 0.05)
            b.ffn.up.weight.normal_(0, 0.05)
    return model


# ---------- untied model ----------

def test_untied_model_has_two_tables():
    cfg = _cfg(tie_word_embeddings=False)
    tied = KSDraftModel(_cfg())
    untied = KSDraftModel(cfg)
    n_tied = sum(p.numel() for p in tied.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    assert untied.head.weight is not untied.embedding.weight
    assert n_untied - n_tied == cfg.vocab_size * cfg.d_model
    assert untied.param_counts()["core"] == cfg.total_params()


def test_head_init_applies_and_tied_rejects():
    cfg = _cfg(tie_word_embeddings=False)
    table = torch.randn(cfg.vocab_size, cfg.d_model)
    model = KSDraftModel(cfg, head_init=table)
    torch.testing.assert_close(model.head.weight.detach(), table)
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        KSDraftModel(_cfg(), head_init=table)


def test_trainer_freezes_both_tables_when_untied():
    torch.manual_seed(0)
    cfg = _cfg(tie_word_embeddings=False)
    model = KSDraftModel(cfg)
    sampler = WindowSampler(torch.randint(0, cfg.vocab_size, (4096,)),
                            seq_len=16, seed=1)
    tcfg = TrainConfig(steps=1, seq_len=16, batch_size=2, val_every=10**6,
                       freeze_embeddings=True, train_dtype="fp32")
    tr = Trainer(model, None, sampler, tcfg, device="cpu",
                 kv_provider=KVFixture(cfg.layers, cfg.kv_dim,
                                       kv_window=cfg.kv_window))
    assert not model.embedding.weight.requires_grad
    assert not model.head.weight.requires_grad


# ---------- export / load ----------

def test_untied_plain_export_stores_both_tables(tmp_path):
    model = _woken(_cfg(tie_word_embeddings=False))
    export_model(model, str(tmp_path), dtype="fp32")
    from safetensors.torch import load_file
    sd = dict(load_file(str(tmp_path / "ksdraft.safetensors")))
    assert "embedding.weight" in sd and "head.weight" in sd
    loaded = load_export(str(tmp_path)).eval()
    assert loaded.head.weight is not loaded.embedding.weight
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    kv = _kv(model, tokens)
    with torch.no_grad():
        y1, _ = model(tokens, target_kv=kv)
        y2, _ = loaded(tokens, target_kv=_kv(loaded, tokens))
    torch.testing.assert_close(y1, y2)


def test_untied_share_vocab_roundtrip(tmp_path):
    cfg = _cfg(tie_word_embeddings=False)
    model = _woken(cfg, seed=1)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    kv = _kv(model, tokens)
    with torch.no_grad():
        y_ref, _ = model(tokens, target_kv=kv)

    export_model(model, str(tmp_path), dtype="fp32", share_vocab=True)
    from safetensors.torch import load_file
    sd = dict(load_file(str(tmp_path / "ksdraft.safetensors")))
    assert "embedding.weight" not in sd and "head.weight" not in sd
    meta = json.loads((tmp_path / "config.json").read_text())
    assert meta["shared_embedding"] and meta["shared_head"]

    with pytest.raises(ValueError, match="head="):
        load_export(str(tmp_path), embedding=model.embedding.weight)
    loaded = load_export(str(tmp_path),
                         embedding=model.embedding.weight.detach().clone(),
                         head=model.head.weight.detach().clone()).eval()
    with torch.no_grad():
        y_new, _ = loaded(tokens, target_kv=_kv(loaded, tokens))
    torch.testing.assert_close(y_ref, y_new)


# ---------- W4 unembed dequant ----------

def _load_init_module():
    path = Path(__file__).resolve().parents[1] / "scripts/init_embedding.py"
    spec = importlib.util.spec_from_file_location("_ksdraft_init", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dequant_unembed_roundtrip():
    """Exact layout check: encode with the fork convention, dequantize, and
    recover the same (code - zero) * scale grid bit-for-bit; plus a coarse
    sanity bound vs the pre-quantization weights."""
    mod = _load_init_module()
    torch.manual_seed(0)
    n, k = 24, 256
    w = (torch.randn(n, k) * 0.05).to(torch.float16)
    groups = k // 128
    w_g = w.float().reshape(n, groups, 128)
    span = w_g.max(-1, keepdim=True).values - w_g.min(-1, keepdim=True).values
    scale = (span / 15.0).clamp_min(1e-6)
    zero = torch.round(-w_g.min(-1, keepdim=True).values / scale)
    codes = torch.round(w_g / scale + zero).clamp(0, 15)
    # fork packing: LOW nibble = even index, HIGH = odd, per byte
    packed = (codes[..., 0::2] + codes[..., 1::2] * 16) \
        .reshape(n, groups, 64).reshape(n, k // 2).to(torch.uint8)
    expected = ((codes - zero) * scale).reshape(n, k).half()
    recovered = mod.dequant_unembed(
        packed, scale.reshape(n, groups).half(),
        zero.reshape(n, groups).half())
    torch.testing.assert_close(recovered.float(), expected.float(),
                               rtol=1e-3, atol=1e-7)
    rel = ((recovered.float() - w.float()).norm() / w.float().norm()).item()
    assert rel < 0.12, rel   # int4 affine grid on a gaussian fixture ~0.10


@pytest.mark.skipif(
    not Path("/home/curved/models/Inkling-2-Mini-PTQR-W4A8-G128").exists(),
    reason="finalized target checkpoint not present")
def test_dequant_real_unembed_rows():
    mod = _load_init_module()
    from safetensors import safe_open
    root = Path("/home/curved/models/Inkling-2-Mini-PTQR-W4A8-G128")
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    shard = root / idx["weight_map"]["model.llm.unembed.weight_scale"]
    with safe_open(str(shard), framework="pt") as f:
        w = f.get_slice("model.llm.unembed.weight")
        sc = f.get_slice("model.llm.unembed.weight_scale")
        zr = f.get_slice("model.llm.unembed.weight_zero")
        rows = mod.dequant_unembed(w[0:64], sc[0:64], zr[0:64])
    assert rows.shape == (64, 2304)
    assert torch.isfinite(rows).all()
    norms = rows.float().norm(dim=-1)
    assert (norms > 0.05).all() and (norms < 20).all(), norms[:8]
