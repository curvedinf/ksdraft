"""DFly2-style target-KV reuse: attention module, block, model, trainer,
gate, and export roundtrip (fixture KV only — no target model needed)."""

import io

import pytest
import torch

from ksdraft.config import KSDraftConfig
from ksdraft.gate import run_gate
from ksdraft.model import KSDraftModel, ReuseBlock, TargetKVAttention, to_int8
from ksdraft.teacher import (FixtureTeacher, KVFixture, causal_kv_windows,
                             fixture_kv_stream)
from ksdraft.train import Trainer, TrainConfig, WindowSampler


def _reuse_cfg(**kw):
    base = dict(vocab_size=256, layers=2, d_model=96, head_size=32,
                rosa_streams=32, ks3_charts=64, kv_dim=64, n_kv_heads=4,
                kv_window=8, dtype="fp32")
    base.update(kw)
    return KSDraftConfig(**base)


def _kv(tokens, layers, kv_dim, window, seed=3):
    k = fixture_kv_stream(tokens, layers, kv_dim, 9_000, seed)
    v = fixture_kv_stream(tokens, layers, kv_dim, 9_100, seed)
    return (causal_kv_windows(k, window).float(),
            causal_kv_windows(v, window).float())


# ---------- config budget ----------

def test_config_reuse_budget_exact():
    cfg = _reuse_cfg()
    assert cfg.q_heads == 3 and cfg.n_kv_eff == 3   # base 3 heads, kv<=3
    assert cfg.reuse_attn_params() == 2 * 96 * 96 + 2 * 64 * 96  # 30,720
    assert cfg.ks3_params() == 64 * (2 * 96 + 6) + 2 * 64        # 12,800
    assert cfg.layer_params() == 43_520
    assert cfg.total_params() == 2 * 43_520 + 256 * 96            # 111,616


def test_config_rosa_default_unchanged():
    plain = KSDraftConfig()
    reuse = KSDraftConfig(kv_dim=1024)
    assert plain.layer_params() == plain.rosa_params() + plain.ks3_params()
    assert plain.total_params() == 24 * plain.layer_params() \
        + 150_144 * 1792
    assert reuse.layer_params() == reuse.reuse_attn_params() + reuse.ks3_params()
    with pytest.raises(ValueError):
        KSDraftConfig(kv_dim=-1)
    with pytest.raises(ValueError):
        KSDraftConfig(n_kv_heads=0)
    with pytest.raises(ValueError):
        KSDraftConfig(kv_window=0)


def test_param_budget_matches_module_tree():
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    counts = model.param_counts()
    assert n == cfg.total_params() + counts["extras"]


# ---------- attention module ----------

def test_attention_matches_bruteforce_gqa():
    torch.manual_seed(1)
    attn = TargetKVAttention(d_model=128, head_size=32, kv_dim=48,
                             n_kv_heads=2)
    with torch.no_grad():  # make the module non-trivial at init
        attn.out_proj.weight.normal_(0, 0.05)
    assert attn.n_q == 4 and attn.n_kv == 2
    B, T, W = 2, 5, 4
    x = torch.randn(B, T, 128)
    k_src = torch.randn(B, T, W, 48)
    v_src = torch.randn(B, T, W, 48)
    out = attn(x, k_src, v_src)
    q = attn.q_proj(x).view(B, T, attn.n_q, attn.head_size)
    k = attn.k_adapter(k_src).view(B, T, W, attn.n_kv, attn.head_size)
    v = attn.v_adapter(v_src).view(B, T, W, attn.n_kv, attn.head_size)
    k = k.repeat_interleave(attn.n_q // attn.n_kv, dim=3)
    v = v.repeat_interleave(attn.n_q // attn.n_kv, dim=3)
    ref = torch.empty(B, T, attn.n_q * attn.head_size)
    for b in range(B):
        for t in range(T):
            scores = torch.einsum("hd,whd->hw", q[b, t], k[b, t]) \
                / attn.head_size ** 0.5
            p = scores.softmax(-1)
            ref[b, t] = torch.einsum("hw,whd->hd", p, v[b, t]).reshape(-1)
    ref = attn.out_proj(ref)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        attn(x, None, v_src)


def test_reuse_block_identity_at_init():
    torch.manual_seed(2)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    kk = torch.randn(2, 6, 4, cfg.kv_dim)
    y, _ = model.blocks[0](x.clone(), None, kk, kk.clone())
    torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)
    assert isinstance(model.blocks[0], ReuseBlock)


def test_forward_requires_kv():
    torch.manual_seed(0)
    model = KSDraftModel(_reuse_cfg())
    tokens = torch.randint(0, 256, (2, 8))
    with pytest.raises(ValueError, match="target_kv"):
        model(tokens)
    with pytest.raises(ValueError, match="target_kv"):
        model(tokens, target_kv=None)


def test_forward_shapes_and_gradient_flow():
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg)
    for block in model.blocks:  # fresh adapters see zero grad through the
        with torch.no_grad():   # zero-init out_proj; open the path first
            block.attn.out_proj.weight.normal_(0, 0.05)
    tokens = torch.randint(0, cfg.vocab_size, (2, 9))
    kv = _kv(tokens[:, :-1], cfg.layers, cfg.kv_dim, cfg.kv_window)
    logits, features = model(tokens[:, :-1], target_kv=kv)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert len(features) == cfg.layers
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    for name in ("q_proj", "k_adapter", "v_adapter", "out_proj"):
        g = model.blocks[0].attn.get_submodule(name).weight.grad
        assert g is not None and g.abs().sum() > 0, name


def test_propose_block_tokens_reuse():
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 9))
    kv = _kv(tokens, cfg.layers, cfg.kv_dim, cfg.kv_window)
    out = model.propose_block_tokens(tokens, 4, topk=8, refine_iters=1,
                                     target_kv=kv)
    assert out.shape == (2, 4)
    assert (out >= 0).all() and (out < cfg.vocab_size).all()


# ---------- fixture KV ----------

def test_causal_windows_reference():
    stream = torch.arange(2 * 3 * 7 * 5, dtype=torch.float32).reshape(2, 3, 7, 5)
    w = causal_kv_windows(stream, 4)
    assert w.shape == (2, 3, 7, 4, 5)
    for t in range(7):
        for j in range(4):
            src = max(0, t - 4 + 1 + j)
            assert torch.equal(w[0, 0, t, j], stream[0, 0, src])


def test_fixture_kv_shapes_and_determinism():
    fx = FixtureTeacher(256, d_tap=16, n_taps=2, topk=4, kv_dim=32,
                        kv_window=4, kv_taps=2, seed=9)
    tokens = torch.randint(0, 256, (2, 6))
    s1, s2 = fx.signals(tokens), fx.signals(tokens)
    for a, b in zip(s1.kv, s2.kv):
        assert a.shape == (2, 2, 6, 4, 32)
        assert a.dtype == torch.float16
        torch.testing.assert_close(a, b)


def test_fixture_kv_is_int8_gs128_grid():
    """Deployed KV is int8_block_g128: the provider emulates that grid, and
    the emulation is a fixed point (re-quantizing changes nothing)."""
    from ksdraft.teacher import causal_kv_windows, emulate_int8_kv, fixture_kv_stream

    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 1024)
    once = emulate_int8_kv(x, 128)
    torch.testing.assert_close(emulate_int8_kv(once, 128), once)

    fx = FixtureTeacher(256, d_tap=16, n_taps=2, topk=4, kv_dim=128,
                        kv_window=4, kv_taps=2, seed=9)
    tokens = torch.randint(0, 256, (2, 6))
    sig = fx.signals(tokens)
    stream = fixture_kv_stream(tokens, 2, 128, 9_000, 9)
    expected = causal_kv_windows(emulate_int8_kv(stream, 128), 4)
    torch.testing.assert_close(sig.kv[0], expected.to(torch.float16))


def test_kv_fixture_provider():
    fx = KVFixture(layers=2, kv_dim=32, kv_window=4, seed=4)
    tokens = torch.randint(0, 256, (3, 5))
    sig = fx.signals(tokens)
    assert sig.kv[0].shape == (3, 2, 5, 4, 32)
    assert sig.states is None
    assert torch.equal(fx.signals(tokens).kv[0], sig.kv[0])


# ---------- trainer (fixture KV, no target model) ----------

def _biased_ids(n, vocab, seed, freq_bias=0.6):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(n, generator=g)
    uni = torch.randint(1, vocab, (n,), generator=g)
    return torch.where(r < freq_bias, torch.zeros(n, dtype=torch.long), uni).tolist()


def test_warmup_with_kv_fixture_runs_and_decreases(tmp_path):
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg)
    sampler = WindowSampler(_biased_ids(20000, 256, seed=1), seq_len=16, seed=11)
    tcfg = TrainConfig(steps=30, seq_len=16, batch_size=4, warmup=5,
                       peak_lr=3e-3, val_every=10**6, train_dtype="fp32")
    provider = KVFixture(cfg.layers, cfg.kv_dim, kv_window=cfg.kv_window, seed=11)
    tr = Trainer(model, None, sampler, tcfg, device="cpu",
                 out_dir=tmp_path, kv_provider=provider)
    log = io.StringIO()
    res = tr.train(stage="warmup", log=lambda m: print(m, file=log))
    assert res["final"]["ce"] < res["initial"]["ce"] - 0.5, res


def test_trainer_reuse_without_kv_source_raises():
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg)
    sampler = WindowSampler(_biased_ids(4096, 256, seed=2), seq_len=16, seed=11)
    tcfg = TrainConfig(steps=2, seq_len=16, batch_size=2, val_every=10**6)
    with pytest.raises(ValueError, match="KV source"):
        Trainer(model, None, sampler, tcfg, device="cpu")


def test_distill_with_fixture_kv_and_block_head(tmp_path):
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    teacher = FixtureTeacher(256, d_tap=16, n_taps=2, topk=4,
                             kv_dim=cfg.kv_dim, kv_window=4,
                             kv_taps=cfg.layers, seed=5)
    model = KSDraftModel(cfg, teacher_dim=2 * 16)
    sampler = WindowSampler(_biased_ids(20000, 256, seed=3), seq_len=16, seed=11)
    tcfg = TrainConfig(steps=12, seq_len=16, batch_size=4, warmup=3,
                       peak_lr=3e-3, val_every=10**6, train_dtype="fp32",
                       block_n=4)
    tr = Trainer(model, teacher, sampler, tcfg, device="cpu",
                 out_dir=tmp_path)
    log = io.StringIO()
    res = tr.train(stage="distill", log=lambda m: print(m, file=log))
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in res["history"])


# ---------- gate / int8 / export ----------

def test_gate_battery_reuse_passes():
    report = run_gate(config=_reuse_cfg(), device="cpu")
    assert report["all_pass"], report["checks"]


def test_to_int8_reuse_forward():
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    kv = _kv(tokens, cfg.layers, cfg.kv_dim, cfg.kv_window)
    quant = to_int8(model).eval()
    with torch.no_grad():
        y, _ = quant(tokens, target_kv=kv)
    assert torch.isfinite(y).all()


def test_export_roundtrip_reuse(tmp_path):
    torch.manual_seed(0)
    from ksdraft.export import export_model, load_export
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    kv = _kv(tokens, cfg.layers, cfg.kv_dim, cfg.kv_window)
    with torch.no_grad():
        y1, _ = model(tokens, target_kv=kv)
    out = export_model(model, str(tmp_path), dtype="fp32")
    assert out.exists()
    loaded = load_export(str(tmp_path), device="cpu").eval()
    assert loaded.config.kv_dim == cfg.kv_dim
    assert loaded.config.n_kv_heads == cfg.n_kv_heads
    with torch.no_grad():
        y2, _ = loaded(tokens, target_kv=kv)
    torch.testing.assert_close(y1, y2)


# ---------- GPU smoke (MI100 / any visible HIP device) ----------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_fp16_autocast_step():
    torch.manual_seed(0)
    cfg = _reuse_cfg()
    model = KSDraftModel(cfg).to("cuda")
    for block in model.blocks:
        with torch.no_grad():
            block.attn.out_proj.weight.normal_(0, 0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, cfg.vocab_size, (8, 9), device="cuda")
    for _ in range(2):
        kv = _kv(tokens[:, :-1].cpu(), cfg.layers, cfg.kv_dim,
                 cfg.kv_window)
        kv = tuple(t.to("cuda", dtype=torch.float16) for t in kv)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(tokens[:, :-1], target_kv=kv)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), tokens[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).all()
