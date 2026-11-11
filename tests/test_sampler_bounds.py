# SPDX-License-Identifier: Apache-2.0
"""Boundary-aware window sampling: windows stay inside a document."""

import torch

from ksdraft.train import WindowSampler, detect_boundaries


def test_windows_never_cross_documents():
    ids = torch.arange(1000)
    bounds = torch.tensor([0, 100, 250, 700])
    s = WindowSampler(ids, seq_len=16, seed=3, boundaries=bounds)
    b = s.batch(64, extra=3)
    assert b.shape == (64, 20)
    # every row starts inside [start, end-span]
    doc_of = torch.searchsorted(bounds, b[:, 0], right=True) - 1
    starts = bounds[doc_of]
    ends = torch.cat([bounds[1:], torch.tensor([1000])])
    assert ((b[:, 0] - starts) >= 0).all()
    assert ((b[:, -1] < ends[doc_of])).all()
    # contiguous
    assert (b[:, 1:] - b[:, :-1] == 1).all()


def test_short_documents_are_skipped():
    ids = torch.arange(50)
    s = WindowSampler(ids, seq_len=16, seed=3, boundaries=[0, 10, 20, 50])
    b = s.batch(32, extra=0)
    # documents are 10, 10 and 30 tokens; span is 17, so only [20,50) fits
    assert (b[:, 0] >= 20).all()


def test_detect_boundaries():
    sep = [10, 11]
    ids = torch.tensor([1, 2, 10, 11, 3, 4, 10, 11, 5])
    assert detect_boundaries(ids, sep).tolist() == [2, 6]
