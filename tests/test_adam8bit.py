"""AdamW8bit: block-wise int8 m and sqrt(v) storage, fp32 math, wiring."""

import io
import math

import pytest
import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model import KSDraftModel
from ksdraft.train import Trainer, TrainConfig, WindowSampler
from ksdraft.train.adam8bit import (AdamW8bit, dequantize_state,
                                    quantize_state)


def _tiny_cfg(**kw):
    base = dict(vocab_size=256, layers=2, d_model=96, head_size=32,
                rosa_streams=32, ks3_charts=64)
    base.update(kw)
    return KSDraftConfig(**base)


# ---------- quantization primitives ----------

def test_quantize_roundtrip_precision():
    torch.manual_seed(0)
    for shape in [(1000,), (7, 333), (2048, 16), (3,)]:
        x = torch.randn(*shape)
        codes, scales = quantize_state(x)
        back = dequantize_state(codes, scales, x.numel(), x)
        assert codes.dtype == torch.int8
        rel = ((back - x).norm() / x.norm()).item()
        assert rel < 0.02, (shape, rel)   # < 1/127 per-block absmax


# ---------- optimizer behavior ----------

def _run(opt_cls, steps=200, **kw):
    torch.manual_seed(7)
    X = torch.randn(512, 64) / 8          # unit-column-norm conditioning
    W = torch.randn(64, 32)
    y = X @ W
    torch.manual_seed(3)
    model = torch.nn.Linear(64, 32, bias=False)
    common = dict(lr=3e-2, weight_decay=0.0, betas=(0.9, 0.99))
    common.update(kw)
    opt = opt_cls(model.parameters(), **common)
    loss = None
    for _ in range(steps):
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, float(loss.detach())


def test_parity_with_fp32_adamw():
    _, l32 = _run(torch.optim.AdamW)
    _, l8 = _run(AdamW8bit)
    initial = 1.05  # both start here (fixed seeds)
    assert l32 < 0.01 * initial and l8 < 0.01 * initial  # both solve it
    assert abs(l8 - l32) < 5e-4 + 0.1 * l32, (l8, l32)


def test_reference_step_math_single_step():
    """One step from zero state is exact fp32: pin the anvil.py form —
    eps on raw sqrt(v), corrections in the step size, cautious lr^2 decay."""
    lr, beta1, beta2, eps = 1e-2, 0.9, 0.99, 1e-8
    p0 = torch.tensor([1.0, -1.0, 1.0, -1.0])
    g = torch.tensor([1.0, 1.0, -1.0, -1.0])
    wd, wd_mul = 0.5, 150.0
    opt = AdamW8bit([p0.clone().requires_grad_(True)], lr=lr,
                    betas=(beta1, beta2), eps=eps, weight_decay=wd)
    opt.param_groups[0]["wd_mul"] = wd_mul
    p = opt.param_groups[0]["params"][0]
    p.grad = g.clone()
    opt.step()
    # hand-computed reference step
    m = (1 - beta1) * g
    s = ((1 - beta2) * g * g).sqrt()
    step_size = lr * math.sqrt(1 - beta2 ** 1) / (1 - beta1 ** 1)
    update = (m / (s + eps)) * step_size
    mask = (update * p0) > 0
    expected = p0 - update - p0 * mask * (lr * lr * wd * wd_mul)
    torch.testing.assert_close(p.detach(), expected, rtol=1e-6, atol=1e-9)
    # the cautious mask is nontrivial on this fixture: decay lands only
    # where the update and the parameter agree in sign
    assert mask.tolist() == [True, False, False, True]


def test_state_memory_is_eighth():
    torch.manual_seed(7)
    X = torch.randn(512, 64) / 8
    y = X @ torch.randn(64, 32)
    torch.manual_seed(3)
    model = torch.nn.Linear(64, 32, bias=False)
    opt = AdamW8bit(model.parameters(), lr=3e-2)
    for _ in range(3):
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    n = sum(p.numel() for p in model.parameters())
    fp32_states = 2 * n * 4
    assert opt.state_bytes() < 0.3 * fp32_states, (opt.state_bytes(), fp32_states)


def test_legacy_fp32_state_load():
    torch.manual_seed(7)
    X = torch.randn(512, 64) / 8
    y = X @ torch.randn(64, 32)
    torch.manual_seed(3)
    model = torch.nn.Linear(64, 32, bias=False)
    opt = AdamW8bit(model.parameters(), lr=3e-2)
    loss = ((model(X) - y) ** 2).mean()
    loss.backward()
    opt.step()
    sd = opt.state_dict()
    # rewrite the states as a legacy fp32 AdamW checkpoint (m, v)
    for _, state in list(sd["state"].items()):
        n = state["m_codes"].numel()
        m = dequantize_state(state["m_codes"], state["m_scales"], n,
                             state["m_codes"].float().view(-1)[:n])
        s = dequantize_state(state["s_codes"], state["s_scales"], n,
                             state["s_codes"].float().view(-1)[:n])
        for key in ("m_codes", "m_scales", "s_codes", "s_scales"):
            state.pop(key)
        state["m"], state["v"] = m.view(-1), (s * s).view(-1)
    opt2 = AdamW8bit(model.parameters(), lr=3e-2)
    opt2.load_state_dict(sd)
    for state in opt2.state.values():
        assert "m_codes" in state and state["m_codes"].dtype == torch.int8
        assert "s_codes" in state and state["s_codes"].dtype == torch.int8
    opt2.step()  # runs after the legacy load


# ---------- trainer wiring ----------

def test_trainer_defaults_to_adam8bit():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = KSDraftModel(cfg)
    ids = torch.randint(0, 256, (4096,))
    sampler = WindowSampler(ids, seq_len=16, seed=1)
    tcfg = TrainConfig(steps=2, seq_len=16, batch_size=2, warmup=1,
                       val_every=10**6, train_dtype="fp32")
    tr = Trainer(model, None, sampler, tcfg, device="cpu")
    assert isinstance(tr.opt, AdamW8bit)
    res = tr.train(stage="warmup", log=lambda m: None)
    assert res["final"]["loss"] == res["final"]["loss"]  # finite


def test_trainer_no_adam8bit_optout():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = KSDraftModel(cfg)
    sampler = WindowSampler(torch.randint(0, 256, (4096,)), seq_len=16, seed=1)
    tcfg = TrainConfig(steps=1, seq_len=16, batch_size=2, val_every=10**6,
                       adam8bit=False, train_dtype="fp32")
    tr = Trainer(model, None, sampler, tcfg, device="cpu")
    assert isinstance(tr.opt, torch.optim.AdamW)
    assert not isinstance(tr.opt, AdamW8bit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="no torch Muon")
def test_trainer_muon_and_adam8bit_on_gpu():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = KSDraftModel(cfg)
    sampler = WindowSampler(torch.randint(0, 256, (8192,)), seq_len=16, seed=1)
    tcfg = TrainConfig(steps=2, seq_len=16, batch_size=4, warmup=1,
                       val_every=10**6, train_dtype="fp16")
    tr = Trainer(model, None, sampler, tcfg, device="cuda")
    assert tr.opt_muon is not None          # split Muon arm active on GPU
    assert isinstance(tr.opt, AdamW8bit)    # 8-bit Adam arm
    log = io.StringIO()
    res = tr.train(stage="warmup", log=lambda m: print(m, file=log))
    assert res["final"]["loss"] == res["final"]["loss"]  # finite


def test_resume_pytorch_adamw_state():
    p = torch.nn.Parameter(torch.ones(16))
    old = torch.optim.AdamW([p])
    p.grad = torch.ones_like(p)
    old.step()
    opt = AdamW8bit([p])
    opt.load_state_dict(old.state_dict())
    assert opt.state[p]["m_codes"].dtype == torch.int8
    opt.step()
    assert opt.state[p]["step"] == 2
