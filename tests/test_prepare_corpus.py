# SPDX-License-Identifier: Apache-2.0
"""prepare_corpus: newest-shard selection, boundary split, val rebasing."""

import subprocess
import sys
from pathlib import Path

import torch


def test_prepare_corpus_splits_by_document(tmp_path: Path):
    tokens = torch.arange(200, dtype=torch.int64)
    bounds = torch.tensor([0, 60, 130], dtype=torch.int64)
    meta = [{"kind": "chat", "prompt": 2, "response": 3}] * 3
    torch.save({"tokens": tokens, "boundaries": bounds, "meta": meta},
               tmp_path / "shard_000003.pt")
    out = tmp_path / "ready"
    subprocess.run(
        [sys.executable, "scripts/prepare_corpus.py", "--shards", str(tmp_path),
         "--out", str(out)],
        check=True, capture_output=True)
    tr = torch.load(out / "train_tokens.pt", weights_only=True)
    va = torch.load(out / "val_tokens.pt", weights_only=True)
    assert tr["tokens"].tolist() == list(range(130))
    assert tr["boundaries"].tolist() == [0, 60]
    assert va["tokens"].tolist() == list(range(130, 200))
    assert va["boundaries"].tolist() == [0]
    assert (out / "summary.json").exists()


def test_prepare_corpus_merges_tiers(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    torch.save({"tokens": torch.arange(100, dtype=torch.int64),
                "boundaries": torch.tensor([0, 50, 90])},
               a / "shard_000002.pt")
    torch.save({"tokens": torch.arange(100, 160, dtype=torch.int64),
                "boundaries": torch.tensor([0, 30])},
               b / "shard_000001.pt")
    out = tmp_path / "merged"
    subprocess.run(
        [sys.executable, "scripts/prepare_corpus.py",
         "--shards", str(a), "--shards", str(b), "--out", str(out)],
        check=True, capture_output=True)
    tr = torch.load(out / "train_tokens.pt", weights_only=True)
    va = torch.load(out / "val_tokens.pt", weights_only=True)
    assert tr["tokens"].numel() == 130
    assert tr["boundaries"].tolist() == [0, 50, 90, 100]
    assert va["tokens"].numel() == 30
    assert va["boundaries"].tolist() == [0]
