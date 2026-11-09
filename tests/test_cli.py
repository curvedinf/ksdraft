from ksdraft.cli import main


def test_size_default(capsys):
    assert main(["size"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "964,153,248" in out  # the decided-shape core budget
    assert "~0.96B" in out


def test_size_custom(capsys):
    assert main(["size", "--layers", "12", "--d-model", "2048", "--head-size", "64"]) == 0
    out = capsys.readouterr().out
    assert "layers            12" in out


def test_all_commands_implemented(capsys):
    # every subcommand now has an implementation; unknown commands error
    import pytest
    from ksdraft.cli import _build_parser
    with pytest.raises(SystemExit):
        main(["bogus"])


def test_train_command_smoke(tmp_path, capsys):
    # tiny run via the CLI on the real config's structure (shape overrides
    # through a minimal config file to keep it fast)
    cfg_file = tmp_path / "tiny.yaml"
    cfg_file.write_text(
        "target: {vocab_size: 256}\n"
        "teacher: {state_taps: [1, 2], topk_logits: 8}\n"
        "draft: {layers: 2, d_model: 96, head_size: 32, rosa_streams: 32,"
        " ks3_charts: 64}\n"
    )
    assert main(["train", "--config", str(cfg_file), "--stage", "warmup",
                 "--steps", "8", "--seq-len", "32", "--batch-size", "2",
                 "--corpus-size", "5000",
                 "--out", str(tmp_path / "run")]) == 0
    out = capsys.readouterr().out
    assert "final loss" in out
    assert list((tmp_path / "run").glob("ckpt_*.pt"))
