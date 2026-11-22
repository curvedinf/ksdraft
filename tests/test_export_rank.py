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


def test_hf_repo_loader_shim(tmp_path):
    import importlib.util
    import torch

    from ksdraft.config import KSDraftConfig
    from ksdraft.export import export_model, load_export
    from ksdraft.model import KSDraftModel

    cfg = KSDraftConfig(vocab_size=128, layers=2, d_model=32, head_size=16,
                        rosa_streams=16, ks3_charts=64)
    model = KSDraftModel(cfg, transfer_space={"C": torch.randn(128, 8),
                                              "rank": 8}).eval()
    out = tmp_path / "repo"
    export_model(model, str(out))
    shim = out / "modeling_ksdraft.py"
    assert shim.exists()
    spec = importlib.util.spec_from_file_location("_modeling_ksdraft", shim)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    draft = mod.from_pretrained(str(out))
    assert draft.spine_core.rank == 8
    tok = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        lg, _ = draft(tok)
    assert torch.isfinite(lg).all()


def test_selector_reads_restored_table_after_dtype_move():
    cfg = KSDraftConfig(vocab_size=128, layers=2, d_model=32, head_size=16,
                        rosa_streams=16, ks3_charts=64)
    model = KSDraftModel(cfg, transfer_space={"rank": 8}).double()
    state = model.state_dict()
    state["transfer_C"] = torch.randn_like(model.transfer_C)
    model.load_state_dict(state)
    for layer in [*(block.injection for block in model.blocks), model.head_write]:
        assert layer.C is model.transfer_C
    assert model.head.weight is model.embedding.weight
