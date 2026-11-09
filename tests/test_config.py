import pytest

from ksdraft.config import KSDraftConfig


def test_decided_shape_budget():
    """Locks the Section 9 decided-shape arithmetic to exact integers."""
    cfg = KSDraftConfig()
    assert (cfg.layers, cfg.d_model, cfg.head_size) == (24, 1536, 64)
    assert cfg.n_heads == 24
    assert cfg.charts == 5888  # round(1536 * 368 / 96)
    assert cfg.rosa_streams == 512

    assert cfg.mixer_params() == 9_950_208  # faithful TimeMix port: 4d^2 + 334d
    assert cfg.rosa_params() == 3_153_920  # 4ds + 4d + 4s (biased projections)
    assert cfg.ks3_params() == 18_135_040
    assert cfg.layer_params() == 31_239_168

    assert cfg.embedding_params() == 150_144 * 1536
    assert cfg.total_params() == 24 * 31_239_168 + 150_144 * 1536
    assert 0.97e9 < cfg.total_params() < 1.0e9  # ~0.98B core (within decided ~1B)


def test_untied_head_doubles_embedding():
    cfg = KSDraftConfig(tie_word_embeddings=False)
    assert cfg.embedding_params() == 2 * 150_144 * 1536


def test_explicit_charts_override():
    assert KSDraftConfig(ks3_charts=1000).charts == 1000


def test_validation():
    with pytest.raises(ValueError):
        KSDraftConfig(dtype="fp8")
    with pytest.raises(ValueError):
        KSDraftConfig(d_model=1000, head_size=64)  # not divisible
    with pytest.raises(ValueError):
        KSDraftConfig(rosa_bits=0)
    with pytest.raises(ValueError):
        KSDraftConfig(rosa_bits=65)


def test_summary_shape():
    s = KSDraftConfig().summary()
    assert s["total_params"] == KSDraftConfig().total_params()
    assert s["dtype"] == "int8"
