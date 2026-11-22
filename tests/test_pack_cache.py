# SPDX-License-Identifier: Apache-2.0
"""Per-step pack caches: pure-function identity + staleness invalidation.

The training forward reuses weight packs within a step (bitwise lossless: a
pack is a pure function of the masters) and must repack after any master
update (optimizer step or checkpoint load).
"""
import torch

from ksdraft.kernels.linear import Int8Linear
from ksdraft.kernels.quant import (PACK_EPOCH, bump_pack_epoch,
                                   quantize_last_dim)


def test_linear_train_pack_reuse_and_invalidation():
    torch.manual_seed(0)
    lin = Int8Linear(256, 128)
    lin.weight.data.normal_(0, 0.02)
    c1, s1 = lin._train_pack()
    ref_c, ref_s = quantize_last_dim(lin.weight.detach().float())
    assert torch.equal(c1, ref_c) and torch.equal(s1.float(), ref_s.float())
    # same epoch: the very same tensors are returned (no repack)
    c2, s2 = lin._train_pack()
    assert c2 is c1 and s2 is s1
    # an optimizer step (epoch bump + master move) must repack
    bump_pack_epoch()
    with torch.no_grad():
        lin.weight.add_(0.01)
    c3, _ = lin._train_pack()
    ref3, _ = quantize_last_dim(lin.weight.detach().float())
    assert not torch.equal(c3, c1)
    assert torch.equal(c3, ref3)


def test_head_cache_epoch_and_version():
    from ksdraft.kernels.fused_train import _quantize_weight_cached

    torch.manual_seed(1)
    w = torch.randn(512, 256)
    c1, s1 = _quantize_weight_cached(w)
    ref_c, ref_s = quantize_last_dim(w.detach().float())
    assert torch.equal(c1, ref_c) and torch.equal(s1.float(), ref_s.float())
    c2, s2 = _quantize_weight_cached(w)
    assert c2 is c1 and s2 is s1
    bump_pack_epoch()
    c3, _ = _quantize_weight_cached(w)
    ref3, _ = quantize_last_dim(w.detach().float())
    assert torch.equal(c3, ref3)


def test_ks3_repack_skips_within_epoch_and_updates_after():
    from ksdraft.model.ks3 import KS3
    from ksdraft.kernels.ks3_int8 import KS3Int8Cell

    torch.manual_seed(2)
    cell = KS3Int8Cell(KS3(128, 8), group=128, trainable=True)
    cell._repack_masters()
    up1 = cell.up_codes.clone()
    fp1 = cell._packed_fp
    cell._repack_masters()          # same epoch+fingerprint: no repack
    assert cell._packed_fp == fp1
    assert cell.up_codes is not None
    bump_pack_epoch()
    with torch.no_grad():
        cell.master_up.add_(0.001)
    cell._repack_masters()
    assert cell._packed_fp != fp1
    ref_c, _ = quantize_last_dim(cell.master_up.detach().float())
    assert torch.equal(cell.up_codes, ref_c)
    assert not torch.equal(up1, cell.up_codes)
