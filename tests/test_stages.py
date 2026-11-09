import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model import KSDraftModel
from ksdraft.teacher import FixtureTeacher
from ksdraft.train import Trainer, TrainConfig, WindowSampler
from ksdraft.train.stages import (
    quantize_last_dim_dithered, wtv_loss, expected_accepted_length,
    ForesightAdapter, fidelity_gate)


def _tiny_cfg():
    return KSDraftConfig(vocab_size=256, layers=2, d_model=96, head_size=32,
                         rosa_streams=32, ks3_charts=64)


def _trainer(stage, steps=12, tmp_path=None):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    teacher = FixtureTeacher(256, d_tap=32, n_taps=3, topk=8)
    model = KSDraftModel(cfg, teacher_dim=3 * 32)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.02)
        for b in model.blocks:
            b.rosa.out_proj.weight.normal_(0, 0.05)
            b.ffn.up.weight.normal_(0, 0.05)
    ids = list(torch.randint(0, 256, (20000,),
                             generator=torch.Generator().manual_seed(1)).tolist())
    sampler = WindowSampler(ids, seq_len=32, seed=11)
    tcfg = TrainConfig(steps=steps, seq_len=32, batch_size=2, warmup=3,
                       peak_lr=3e-3, val_every=10**6)
    return Trainer(model, teacher, sampler, tcfg, device="cpu",
                   out_dir=tmp_path)


def test_dithered_quantization_contract():
    from ksdraft.kernels.quant import quantize_last_dim, dequantize_last_dim
    x = torch.randn(8, 256) * 3
    codes0, scales0 = quantize_last_dim_dithered(x, 128, dither=0.0)
    codes_e, scales_e = quantize_last_dim(x, 128)
    assert torch.equal(codes0, codes_e)          # dither=0 is the exact grid
    assert torch.equal(scales0, scales_e)
    codes_d, _ = quantize_last_dim_dithered(x, 128, dither=0.5)
    changed = (codes_d != codes_e).float().mean().item()
    assert 0.0 < changed < 0.6                    # explores neighbors, bounded


def test_wtv_and_eal():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 64)
    probs, ids = torch.softmax(torch.randn(2, 8, 64), -1).topk(8, -1)
    loss = wtv_loss(logits, probs, ids)
    assert torch.isfinite(loss) and loss >= 0
    # perfect match -> ~0 WTV
    perfect = torch.full((2, 8, 64), -30.0)
    perfect.scatter_(-1, ids, (probs / probs.sum(-1, keepdim=True)).log())
    assert wtv_loss(perfect, probs, ids).item() < 1e-3
    eal = expected_accepted_length(perfect, torch.zeros(2, 8, dtype=torch.long),
                                   probs, ids)
    assert eal == 8.0  # greedy agrees everywhere


def test_foresight_stage():
    # the fixture teacher's states are unpredictable hashes, so only the
    # anchor-token CE carries learnable signal — assert decrease without margin
    tr = _trainer("foresight", steps=40)
    tr.cfg.peak_lr = 1e-2
    res = tr.train("foresight")
    assert res["final"]["loss"] < res["initial"]["loss"], res
    assert res["final"]["ce"] < res["initial"]["ce"], res
    assert isinstance(res["adapter"], ForesightAdapter)


def test_refine_and_acceptance_stages():
    for stage in ("refine", "acceptance"):
        tr = _trainer(stage)
        res = tr.train(stage)
        assert all(torch.isfinite(torch.tensor(h["loss"])) for h in res["history"])


def test_ptqr_stage_with_fidelity_gate():
    tr = _trainer("ptqr", steps=8)
    res = tr.train("ptqr")
    assert res["history"][-1]["dither"] < res["history"][0]["dither"]
    fid = res["fidelity"]
    assert "top1_mismatch" in fid and "all_pass" in fid


def _woken(cfg, seed, std=0.3):
    torch.manual_seed(seed)
    m = KSDraftModel(cfg).eval()
    with torch.no_grad():
        m.head.weight.normal_(0, 0.05)
        for b in m.blocks:
            b.rosa.out_proj.weight.normal_(0, std)
            b.ffn.up.weight.normal_(0, std)
    return m


def test_fidelity_gate_detects_mismatch():
    # tied-head identity models are copiers (self-dot argmax); wake the block
    # surfaces so real differences exist between the two models
    cfg = _tiny_cfg()
    a = _woken(cfg, 0)
    b = _woken(cfg, 7)
    tokens = torch.randint(0, 256, (2, 8))
    report = fidelity_gate(a, b, tokens)
    assert report["top1_pass"] is False  # mismatch detected
    report_same = fidelity_gate(a, a, tokens)
    assert report_same["top1_pass"] is True
