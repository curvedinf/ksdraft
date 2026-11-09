from ksdraft.cli import main


def test_size_default(capsys):
    assert main(["size"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "1,029,611,520" in out  # the decided ~1.03B budget
    assert "~1.03B" in out


def test_size_custom(capsys):
    assert main(["size", "--layers", "12", "--d-model", "2048", "--head-size", "64"]) == 0
    out = capsys.readouterr().out
    assert "layers            12" in out


def test_stub_commands_return_error(capsys):
    for cmd in ("mine", "train", "gate", "export"):
        assert main([cmd]) == 2
