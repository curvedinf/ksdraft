# SPDX-License-Identifier: Apache-2.0
"""Exports carry the DSpine rank and transfer table into load_export."""

import torch

from ksdraft.config import KSDraftConfig
from ksdraft.export import export_model, load_export
from ksdraft.model import KSDraftModel


def test_export_preserves_transfer_rank(tmp_path):
    torch.manual_seed(0)
    cfg = KSDraftConfig(vocab_size=128, layers=2, d_model=32, head_size=16,
                        rosa_streams=16, ks3_charts=64)
    ts = {"C": torch.randn(cfg.vocab_size, 8), "rank": 8}
    model = KSDraftModel(cfg, transfer_space=ts).eval()
    out = tmp_path / "exp"
    export_model(model, str(out))
    loaded = load_export(str(out), device="cpu").eval()
    assert loaded.spine_core.rank == 8
    assert loaded.transfer_C.shape == (cfg.vocab_size, 8)
    assert torch.allclose(loaded.transfer_C.float(),
                          model.transfer_C.float(), atol=1e-3)
    tok = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        lg, _ = loaded(tok)
    assert torch.isfinite(lg).all()
