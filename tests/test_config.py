import pytest

from ksdraft.config import KSDraftConfig


def test_decided_shape_budget():
    """Locks the Section 9 decided-shape arithmetic (RWKV-8-only backbone,
    corrected 2026-11-08) to exact integers."""
    cfg = KSDraftConfig()
    assert (cfg.layers, cfg.d_model, cfg.head_size) == (24, 1792, 64)
    assert cfg.n_heads == 28
    assert cfg.charts == 6869  # round(1792 * 368 / 96)
    assert cfg.rosa_streams == 597  # round(1792 * 32 / 96)

    assert cfg.rosa_params() == 4_288_852  # 4ds + 4d + 4s (biased projections)
    assert cfg.ks3_params() == 24_673_448
    assert cfg.layer_params() == 28_962_300

    assert cfg.embedding_params() == 150_144 * 1792
    assert cfg.total_params() == 24 * 28_962_300 + 150_144 * 1792
    assert 0.95e9 < cfg.total_params() < 0.98e9  # ~0.96B core (~1B envelope)


def test_untied_head_doubles_embedding():
    cfg = KSDraftConfig(tie_word_embeddings=False)
    assert cfg.embedding_params() == 2 * 150_144 * 1792


def test_explicit_charts_override():
    assert KSDraftConfig(ks3_charts=1000).charts == 1000


def test_validation():
    with pytest.raises(ValueError):
        KSDraftConfig(dtype="fp8")
    with pytest.raises(ValueError):
        KSDraftConfig(rosa_bits=0)
    with pytest.raises(ValueError):
        KSDraftConfig(rosa_bits=65)


def test_summary_shape():
    s = KSDraftConfig().summary()
    assert s["total_params"] == KSDraftConfig().total_params()
    assert s["dtype"] == "int8"
