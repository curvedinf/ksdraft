# SPDX-License-Identifier: Apache-2.0
"""INT8 cache-form KV path: score-math parity with the dequantized reference.

The attention out_proj is zero-initialized by design, so these tests first
randomize it — otherwise every attention-output comparison is trivially
equal. The codes path must (a) stay close to the dequantized reference and
(b) be causal per query with grouped heads (a token-major/head-major
reshape slip would leak future rows into earlier queries).
"""
import pytest
import torch

from ksdraft.config import KSDraftConfig
from ksdraft.model import KSDraftModel, to_int8
from ksdraft.teacher import blocks_int8_kv, dequant_kv


def _model():
    cfg = KSDraftConfig(vocab_size=4096, layers=1, d_model=2048,
                        head_size=64, rosa_streams=683, ks3_charts=1024,
                        kv_dim=1024, n_kv_heads=8, kv_window=1024,
                        dtype="int8", tie_word_embeddings=False)
    torch.manual_seed(0)
    m = to_int8(KSDraftModel(cfg)).eval()
    with torch.no_grad():
        # make the attention path observable (ctor zero-inits out_proj)
        m.blocks[0].attn.out_proj.weight.normal_(0, 0.05)
    return m


def _kv(B, taps, S):
    torch.manual_seed(1)
    k = torch.randn(B, taps, S, 1024)
    v = torch.randn(B, taps, S, 1024)
    kc, ks = blocks_int8_kv(k)
    vc, vs = blocks_int8_kv(v)
    return (kc, ks, vc, vs), (dequant_kv(kc, ks), dequant_kv(vc, vs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_codes_path_matches_dequant_reference():
    m = _model().cuda()
    B, T, W = 2, 16, 64
    S = W + T - 1
    codes, dq = _kv(B, 1, S)
    toks = torch.randint(0, 4096, (B, T), device="cuda")
    with torch.no_grad():
        lo_ref, _ = m(toks, target_kv=tuple(t.cuda() for t in dq))
        lo_i8, _ = m(toks, target_kv=tuple(t.cuda() for t in codes))
    d = (lo_ref - lo_i8).abs()
    agree = (lo_ref.argmax(-1) == lo_i8.argmax(-1)).float().mean()
    assert d.max() < 0.5, f"int8 score path diverges too far: {d.max()}"
    assert agree > 0.95, f"top-1 agreement {agree}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_codes_path_is_causal_per_query():
    m = _model().cuda()
    B, T, W = 1, 16, 64
    S = W + T - 1
    off = S - T
    codes, _ = _kv(B, 1, S)
    toks = torch.randint(0, 4096, (B, T), device="cuda")
    kc, ks, vc, vs = [t.cuda() for t in codes]
    with torch.no_grad():
        base, _ = m(toks, target_kv=(kc, ks, vc, vs))
    # perturb the K/V rows at stream position off+5: future for queries t<5
    p = off + 5
    kc2 = kc.clone()
    kc2[:, :, p] = -kc2[:, :, p]
    vc2 = vc.clone()
    vc2[:, :, p] = -vc2[:, :, p]
    with torch.no_grad():
        pert, _ = m(toks, target_kv=(kc2, ks, vc2, vs))
    early = (base[:, :5] - pert[:, :5]).abs().max().item()
    late = (base[:, 5:] - pert[:, 5:]).abs().max().item()
    assert early < 1e-5, f"future KV leaked into earlier queries: {early}"
    assert late > 1e-3, f"perturbation had no effect (test vacuous): {late}"
