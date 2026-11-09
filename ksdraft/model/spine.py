"""DSpine-style gated adjacent injection + per-layer transfer-space features.

Mechanism (DSpine, arXiv 2609.36173): at every layer, each position receives
its predecessor's *predicted feature*, gated by both the current state and the
injected feature, so the causal conditioning chain unfolds over network depth
while positions still update in one pass:

    p_t   = F(z_t)                     # predicted feature (transfer space)
    g_t   = sigmoid(G_q z_t + G_p p_{t-1} + b)
    z'_t  = z_t + g_t * (A p_{t-1})

`p_t` is exposed per layer for the transfer-space auxiliary loss (align with
the embedding of the correct next token — DSpine's unified transfer space).

All gates/injections are low-rank so the mechanism stays a small fraction of
the layer budget (configurable; defaults add ~0.5M params/layer at d=1536,
~1% of the total, not counted in the core Section 9 formulas).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _LowRank(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class AdjacentInjection(nn.Module):
    def __init__(self, width: int, rank: int = 32, feature_rank: int = 64):
        super().__init__()
        self.width = width
        self.feature_down = nn.Linear(width, feature_rank, bias=False)
        self.feature_up = nn.Linear(feature_rank, width, bias=False)
        self.gate_q = _LowRank(width, rank)
        self.gate_p = _LowRank(width, rank)
        self.gate_bias = nn.Parameter(torch.zeros(width))
        self.inject = _LowRank(width, rank)

    def extra_repr(self) -> str:
        return f"width={self.width}"

    def predicted_feature(self, z: torch.Tensor) -> torch.Tensor:
        """p_t: this layer's prediction of the next token's feature."""
        return self.feature_up(self.feature_down(z))

    def forward(
        self, z: torch.Tensor, prev_layer_feature: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z', p) where p is this layer's predicted feature (for the
        transfer-space loss). `prev_layer_feature`, when given, seeds position
        0's injected feature from the previous layer's prediction."""
        p = self.predicted_feature(z)
        p_prev = F.pad(p, (0, 0, 1, 0))[:, :-1, :]
        if prev_layer_feature is not None:
            p_prev = torch.cat([prev_layer_feature[:, :1, :], p_prev[:, 1:, :]], dim=1)
        gate = torch.sigmoid(
            self.gate_q(z) + self.gate_p(p_prev) + self.gate_bias)
        return z + gate * self.inject(p_prev), p
