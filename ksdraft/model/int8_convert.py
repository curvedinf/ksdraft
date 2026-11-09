"""Model-wide INT8 conversion (RESEARCH.md Sections 8.2 and 12).

``to_int8`` returns a converted copy of a KSDraftModel for the int8 tier:

- GEMM surfaces -> ``Int8Linear`` (W8A8 GS128; fp32 masters stay
  authoritative; STE autograd in training, W8A8 kernels in eval)
- KS3 decoders -> ``KS3Int8Cell`` (all-INT8 runtime; int8 nonlinearities)
- **Float exceptions per the fork's doctrine** (Section 8.2, measured-not-
  assumed): LayerNorms, Rosa payload + time-mix vectors, DSpine injection
  gates (small low-rank surfaces), and the embedding/head (gather-heavy;
  int8-embedding with per-row scales is a separate staged surface, as in
  the fork's generation-by-generation adoption).

Every dtype change on these surfaces is acceptance-gated before adoption
(``ksdraft gate``); this conversion provides the candidate.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from ..config import KSDraftConfig
from ..kernels.linear import Int8Linear
from ..kernels.ks3_int8 import KS3Int8Cell
from .draft import KSDraftModel


def to_int8(model: KSDraftModel, group: int = 128,
            nonlinear: str = "int8") -> KSDraftModel:
    converted = copy.deepcopy(model)
    n_swapped = 0
    for block in converted.blocks:
        rosa = block.rosa
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(rosa, name, Int8Linear.from_float(getattr(rosa, name)))
            n_swapped += 1
        block.ffn = KS3Int8Cell(cell=block.ffn, group=group, nonlinear=nonlinear)
        n_swapped += 1
    if converted.teacher_proj is not None:
        converted.teacher_proj = Int8Linear.from_float(converted.teacher_proj)
        n_swapped += 1
    converted._int8_tier = {"group": group, "nonlinear": nonlinear,
                            "swapped_surfaces": n_swapped}
    return converted


def int8_surfaces(converted: KSDraftModel) -> dict:
    info = getattr(converted, "_int8_tier", None)
    return dict(info) if info else {}
