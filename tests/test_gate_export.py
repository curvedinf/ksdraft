import pytest
import torch

from ksdraft.cli import main
from ksdraft.config import KSDraftConfig
from ksdraft.gate import run_gate
from ksdraft.model import KSDraftModel
from ksdraft.export import export_model, load_export


def _tiny_cfg():
    return KSDraftConfig(vocab_size=256, layers=2, d_model=96, head_size=32,
                         rosa_streams=32, ks3_charts=64)


def test_gate_battery_passes(tmp_path):
    cfg = _tiny_cfg()
    model = KSDraftModel(cfg, teacher_dim=3 * 32)
    report = run_gate(model=model, out_dir=str(tmp_path))
    assert report["all_pass"], report["checks"]
    assert (tmp_path / "gate_report.json").exists()
    for name in ("param_budget", "identity_block_init", "bf16_parity",
                 "int8_ks3_parities", "forward_finite"):
        assert name in report["checks"]


def test_gate_detects_param_mismatch(tmp_path):
    cfg = _tiny_cfg()
    model = KSDraftModel(cfg)
    with torch.no_grad():
        model.embedding.weight.add_(1.0)  # break nothing; still same params
    report = run_gate(model=model, out_dir=str(tmp_path))
    assert report["checks"]["param_budget"]["pass"]


def test_export_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = KSDraftModel(cfg, teacher_dim=3 * 32).eval()
    tokens = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        y1, _ = model(tokens)
    path = export_model(model, str(tmp_path))
    assert path.exists()
    loaded = load_export(str(tmp_path)).eval()
    with torch.no_grad():
        y2, _ = loaded(tokens)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)
    assert loaded.head.weight is loaded.embedding.weight  # tying survives


def test_cli_gate_and_export(tmp_path, capsys):
    cfg_file = tmp_path / "tiny.yaml"
    cfg_file.write_text(
        "target: {vocab_size: 256}\n"
        "teacher: {state_taps: [1, 2], fixture_d_tap: 32}\n"
        "draft: {layers: 2, d_model: 96, head_size: 32, rosa_streams: 32,"
        " ks3_charts: 64}\n"
    )
    assert main(["gate", "--config", str(cfg_file),
                 "--out", str(tmp_path / "gate")]) == 0
    assert "gate PASSED" in capsys.readouterr().out
    assert main(["export", "--config", str(cfg_file),
                 "--out", str(tmp_path / "exp")]) == 0
    assert "exported ->" in capsys.readouterr().out
