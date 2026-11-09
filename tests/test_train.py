import io
import sys
from pathlib import Path

import pytest
import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model import KSDraftModel
from ksdraft.teacher import FixtureTeacher
from ksdraft.data import CorpusConfig, build_corpus, train_val_split, _needle_document
from ksdraft.train import Trainer, TrainConfig, WindowSampler, distillation_losses


def _tiny_model(**kw):
    base = dict(vocab_size=256, layers=2, d_model=96, head_size=32,
                rosa_streams=32, ks3_charts=64)
    base.update(kw)
    return KSDraftConfig(**base)


# ---------- teacher ----------

def test_fixture_teacher_deterministic_and_valid():
    t = FixtureTeacher(vocab_size=256, d_tap=32, n_taps=3, topk=8, seed=5)
    tokens = torch.randint(0, 256, (2, 16))
    s1 = t.signals(tokens)
    s2 = t.signals(tokens)
    assert s1.states.shape == (2, 16, 3, 32)
    assert s1.states.dtype == torch.float16
    assert torch.equal(s1.states, s2.states)          # deterministic
    assert torch.equal(s1.topk_ids, s2.topk_ids)
    sums = s1.topk_probs.float().sum(-1)
    assert (sums > 0).all() and (sums <= 1.0 + 1e-4).all()  # sparse support;
    # the trainer renormalizes this to a simplex (top-k CE on teacher support)
    p = s1.topk_probs.float()
    assert (p[:, :, :-1] >= p[:, :, 1:]).all()         # sorted descending


# ---------- corpus / tier-5 NIAH ----------

def test_needle_document_strata_and_format():
    import random
    rng = random.Random(7)
    docs = [f"paragraph {i} " + "x" * 2000 for i in range(50)]
    hay_uniform, proj1, code1 = _needle_document(rng, docs, index=1)  # uniform depth
    hay_recent, proj2, code2 = _needle_document(rng, docs, index=0)   # last-16k stratum
    for hay, proj, code in ((hay_uniform, proj1, code1), (hay_recent, proj2, code2)):
        assert proj.startswith("PROJECT-") and code.startswith("ZK")
        assert f"The access code for {proj} is {code}." in hay
    # recent-stratum needle sits in the final 16k*3 chars (with slack)
    pos = hay_recent.rfind(f"The access code for {proj2}")
    assert pos > len(hay_recent) - 16 * 1024 * 3 - 64
    pos_u = hay_uniform.rfind(f"The access code for {proj1}")
    assert pos_u <= len(hay_uniform) - 16 * 1024 * 3 + 64


def test_build_corpus_splits_and_sizes():
    import random
    rng = random.Random(3)
    docs = [f"doc {i} " + "w" * 400 for i in range(20)]
    traces = [f"<tool>call {i}</tool> ok" for i in range(10)]
    tok = lambda s: [ord(c) % 256 for c in s][:4096]
    cfg = CorpusConfig(n_c4=10, n_agent=5, n_long=2, n_xlong=1, max_sample=256)
    ids = build_corpus(docs, traces, tok, cfg, rng=rng)
    assert len(ids) > 0 and all(0 <= i < 256 for i in ids)
    tr, va = train_val_split(ids, cfg.val_frac, rng)
    assert len(tr) + len(va) == len(ids) and len(va) >= 1


# ---------- trainer ----------

def _biased_ids(n, vocab, seed, freq_bias=0.6):
    """Learnable synthetic stream: `freq_bias` of tokens are 0 (unigram
    structure the tied embedding/head can fit); the rest uniform."""
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(n, generator=g)
    uni = torch.randint(1, vocab, (n,), generator=g)
    return torch.where(r < freq_bias, torch.zeros(n, dtype=torch.long), uni).tolist()


def _run_trainer(stage, steps=30, tmp_path=None):
    torch.manual_seed(0)
    cfg = _tiny_model()
    teacher = FixtureTeacher(256, d_tap=32, n_taps=3, topk=8) \
        if stage == "distill" else None
    model = KSDraftModel(cfg, teacher_dim=3 * 32 if stage == "distill" else 0)
    ids = _biased_ids(20000, 256, seed=1)
    sampler = WindowSampler(ids, seq_len=32, seed=11)
    tcfg = TrainConfig(steps=steps, seq_len=32, batch_size=4, warmup=5,
                       peak_lr=3e-3, val_every=10**6)
    tr = Trainer(model, teacher, sampler, tcfg, device="cpu", out_dir=tmp_path)
    log = io.StringIO()
    res = tr.train(stage=stage, log=lambda m: print(m, file=log))
    return tr, res


def test_warmup_loss_decreases():
    _, res = _run_trainer("warmup", steps=30)
    assert res["final"]["ce"] < res["initial"]["ce"] - 1.0, res
    assert res["final"]["ce"] < 4.5, res  # well below ln(256)=5.545


def test_distill_with_fixture_teacher_runs_and_decreases():
    _, res = _run_trainer("distill", steps=30)
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in res["history"])
    assert res["final"]["ce"] < res["initial"]["ce"] - 0.5, res


def test_checkpoint_roundtrip(tmp_path):
    tr, res = _run_trainer("warmup", steps=12, tmp_path=tmp_path)
    ckpts = sorted(Path(tmp_path).glob("ckpt_*.pt"))
    assert ckpts
    x = torch.randint(0, 256, (2, 8))
    tr.model.eval()
    with torch.no_grad():
        y1, _ = tr.model(x)
    tr2 = Trainer(tr.model, None, tr.sampler, tr.cfg, out_dir=tmp_path)
    tr2.load(ckpts[-1])
    with torch.no_grad():
        y2, _ = tr2.model(x)
    torch.testing.assert_close(y1, y2)
