"""KS3 spine decoder — reference implementation.

Ported from the K-Splanifolds KS3 specification (curvedinf/k-splanifolds,
experiments/common/ks3.py structure; formulas cross-checked against the fused
kernel math in experiments/common/ks3_int8.py):

    z    = down(x)                        # chart coordinates (rank-1: mean == z)
    t    = 4 * tanh(z * 0.25)             # bounded longitudinal coordinate
    zw   = tanh(softplus(raw_scale) * (t + 0.5*tanh(raw_shift)))
    poly  = ((a*t + b)*t + c2)*t + c0     # bounded Hermite cubic
             a  =  2*c0 - 2*c1 + c2 + c3
             b  = -3*c0 + 3*c1 - 2*c2 - c3
    warp  = ((4*c5*zw + 2*c4)*zw - 3*c5)*zw - c4   # Chebyshev T2/T3 warp
    h    = relu(z + poly + warp) ** 2     # coordinate bypass + ReLU-squared
    y    = up(h)

Six learned coefficients per output coordinate; warp scale/shift per chart.
Parameter count: charts * (2*d + 8) — exactly KSDraftConfig.ks3_params().

Initialization follows the repo doctrine: the output projection starts at zero
so a freshly initialized KS3 contributes nothing (identity-block start).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class KS3(nn.Module):
    def __init__(self, d_model: int, num_charts: int | None = None):
        super().__init__()
        if num_charts is None:
            num_charts = round(d_model * 368 / 96)
        self.d_model = d_model
        self.num_charts = num_charts

        self.down = nn.Linear(d_model, num_charts, bias=False)
        self.up = nn.Linear(num_charts, d_model, bias=False)
        self.coefficients = nn.Parameter(torch.zeros(num_charts, 6))
        self.raw_scale = nn.Parameter(torch.zeros(num_charts))
        self.raw_shift = nn.Parameter(torch.zeros(num_charts))

        nn.init.normal_(self.down.weight, std=d_model**-0.5)
        nn.init.zeros_(self.up.weight)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_charts={self.num_charts}"

    def warp_scale(self) -> torch.Tensor:
        # softplus bound per the stabilized export; matches KS3Int8 packing.
        return F.softplus(self.raw_scale) + 1e-4

    def warp_shift(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.raw_shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        t = 4.0 * torch.tanh(z * 0.25)
        zw = torch.tanh(self.warp_scale() * (t + self.warp_shift()))

        c0, c1, c2, c3, c4, c5 = self.coefficients.unbind(-1)
        a = 2 * c0 - 2 * c1 + c2 + c3
        b = -3 * c0 + 3 * c1 - 2 * c2 - c3
        poly = ((a * t + b) * t + c2) * t + c0
        warp = ((4 * c5 * zw + 2 * c4) * zw - 3 * c5) * zw - c4

        h = F.relu(z + poly + warp).square()
        return self.up(h)
