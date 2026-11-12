# SPDX-License-Identifier: Apache-2.0
"""HTTP teacher parsing (no network)."""

import torch

from ksdraft.teacher.http import HTTPTeacher, parse_entry


def test_parse_entry_truncates_and_renormalizes():
    entry = {str(i): {"logprob": -0.1 * i} for i in range(1, 26)}
    idx, p = parse_entry(entry, topk=32, serve_top_k=20, serve_top_p=0.95)
    assert idx.numel() <= 20
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert idx[0].item() == 1  # best logprob first
    assert torch.all(p[:-1] >= p[1:])  # sorted descending


def test_parse_entry_missing_position():
    assert parse_entry(None, 32, 20, 0.95) is None
    assert parse_entry({}, 32, 20, 0.95) is None


def test_http_teacher_constructor():
    t = HTTPTeacher("http://127.0.0.1:8030/v1", api_key="x", topk=16)
    assert t.topk == 16 and t.n_taps == 0
