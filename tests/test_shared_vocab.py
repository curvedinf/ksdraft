"""Shared-embedding (DFly2 pattern): trunk-only export + target-table bind,
plus the verbatim-copy init path."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ksdraft.config import KSDraftConfig
from ksdraft.model import KSDraftModel
from ksdraft.teacher import KVFixture
from ksdraft.export import export_model, load_export


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
        for b in model.blocks:
            if hasattr(b, "attn"):
                b.attn.out_proj.weight.normal_(0, 0.05)
            b.ffn.up.weight.normal_(0, 0.05)
    return model


def test_share_vocab_roundtrip(tmp_path):
    cfg = _cfg()
    model = _woken(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    kv = _kv(model, tokens)
    with torch.no_grad():
        y_ref, _ = model(tokens, target_kv=kv)

    out = export_model(model, str(tmp_path), dtype="fp32", share_vocab=True)
    assert out.exists()
    from safetensors.torch import load_file
    sd = dict(load_file(str(out)))
    assert "embedding.weight" not in sd and "head.weight" not in sd
    meta = json.loads((tmp_path / "config.json").read_text())
    assert meta["shared_embedding"] is True

    with pytest.raises(ValueError, match="shared_embedding"):
        load_export(str(tmp_path))
    table = model.embedding.weight.detach().clone()
    loaded = load_export(str(tmp_path), embedding=table).eval()
    assert loaded.head.weight is loaded.embedding.weight  # tie survives
    with torch.no_grad():
        y_new, _ = loaded(tokens, target_kv=_kv(loaded, tokens))
    torch.testing.assert_close(y_ref, y_new)


def test_share_vocab_rejects_wrong_shape(tmp_path):
    cfg = _cfg()
    model = _woken(cfg)
    export_model(model, str(tmp_path), dtype="fp32", share_vocab=True)
    bad = torch.zeros(cfg.vocab_size + 1, cfg.d_model)
    with pytest.raises(ValueError, match="does not match"):
        load_export(str(tmp_path), embedding=bad)


def test_plain_export_still_carries_table(tmp_path):
    cfg = _cfg()
    model = _woken(cfg)
    export_model(model, str(tmp_path), dtype="fp32", share_vocab=False)
    from safetensors.torch import load_file
    sd = dict(load_file(str(tmp_path / "ksdraft.safetensors")))
    assert "embedding.weight" in sd
    meta = json.loads((tmp_path / "config.json").read_text())
    assert "shared_embedding" not in meta


def test_init_embedding_copy_mode(tmp_path):
    vocab, d = 64, 96
    rows = torch.randn(vocab, d)
    target = tmp_path / "target"
    target.mkdir()
    save_file({"model.embed_tokens.weight": rows.half()},
              str(target / "model.safetensors"))
    out = tmp_path / "embedding_init.pt"
    script = Path(__file__).resolve().parents[1] / "scripts/init_embedding.py"
    r = subprocess.run(
        [sys.executable, str(script), "--copy", "--device", "cpu",
         "--target", str(target), "--d-model", str(d), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = torch.load(out, weights_only=True)
    torch.testing.assert_close(got.float(), rows, rtol=1e-3, atol=1e-3)

    r_bad = subprocess.run(
        [sys.executable, str(script), "--copy", "--device", "cpu",
         "--target", str(target), "--d-model", "128", "--out", str(out)],
        capture_output=True, text=True)
    assert r_bad.returncode != 0
    assert "equal the target hidden size" in r_bad.stderr + r_bad.stdout
