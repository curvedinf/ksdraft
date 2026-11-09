"""KS3 spine decoder — reference implementation (canonical Horner port).

Ported from curvedinf/k-splanifolds experiments/common/ks3.py:

- Coefficients stay in the Hermite basis; **Horner evaluation changes the
  arithmetic, not the parameterization**. The cubic is evaluated with
  precombined coefficients in nested form, and the warp as p4*T2(z)+p5*T3(z)
  in nested form — matching the fused W8A8 kernels exactly, so the INT8 tier
  and this reference share one contract.
- Geometry (spine evaluation) always runs FP32 (FP64 for double models),
  regardless of input dtype; coordinates clamp to +-65504; projections follow
  normal dtype behavior.
- For rank-1 charts the chart mean is the coordinate itself; higher ranks
  average the chart's coordinates (canonical API; the decided ksdraft shape
  is rank 1).

`spine_basis()` provides the explicit Hermite/Chebyshev-basis oracle used by
the tests to lock the Horner path (the same fused-vs-basis check doctrine as
the k-splanifolds operator tests).

Initialization: canonical ks3.py init (coefficients ~ N(0, 0.05), raw_scale
0.75, raw_shift 0) with the ksdraft addition of a zero-init output
projection, so a fresh KS3 still contributes nothing (identity-block start).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def spine_horner(
    t: torch.Tensor, z: torch.Tensor, coefficients: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Horner-form spine evaluation.

    t: bounded longitudinal coordinate (..., charts, 1)
    z: warp coordinate (..., charts, 1)
    coefficients: (..., charts, 6) — p0..p3 Hermite basis, p4/p5 warp.
    Returns (spine, warped).
    """
    p0, p1, p2, p3, p4, p5 = coefficients.unbind(-1)
    cubic = 2.0 * p0 - 2.0 * p1 + p2 + p3
    quadratic = -3.0 * p0 + 3.0 * p1 - 2.0 * p2 - p3
    spine = ((cubic * t + quadratic) * t + p2) * t + p0          # Horner
    warped = ((4.0 * p5 * z + 2.0 * p4) * z - 3.0 * p5) * z - p4  # Horner
    return spine, warped


def spine_basis(
    t: torch.Tensor, z: torch.Tensor, coefficients: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit-basis oracle for the Horner path (tests only).

    Hermite cubics on the bounded coordinate t:
        h00 = 2t^3 - 3t^2 + 1,  h10 = t^3 - 2t^2 + t,
        h01 = -2t^3 + 3t^2,     h11 = t^3 - t^2
    Chebyshev warp terms: T2(z) = 2z^2 - 1, T3(z) = 4z^3 - 3z.
    """
    p0, p1, p2, p3, p4, p5 = coefficients.unbind(-1)
    t2, t3 = t * t, t * t * t
    spine = (p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2)
             + p2 * (t3 - 2 * t2 + t) + p3 * (t3 - t2))
    warped = p4 * (2 * z * z - 1) + p5 * (4 * z * z * z - 3 * z)
    return spine, warped


class KS3(nn.Module):
    def __init__(self, d_model: int, num_charts: int | None = None,
                 subspace_rank: int = 1):
        super().__init__()
        if num_charts is None:
            num_charts = round(d_model * 368 / 96)
        self.d_model = d_model
        self.num_charts = num_charts
        self.subspace_rank = subspace_rank
        self.features = num_charts * subspace_rank

        self.down = nn.Linear(d_model, self.features, bias=False)
        self.up = nn.Linear(self.features, d_model, bias=False)
        shape = ((num_charts, 6) if subspace_rank == 1
                 else (num_charts, subspace_rank, 6))
        self.coefficients = nn.Parameter(torch.empty(shape))
        self.raw_scale = nn.Parameter(torch.full((num_charts,), 0.75))
        self.raw_shift = nn.Parameter(torch.zeros(num_charts))

        nn.init.normal_(self.coefficients, std=0.05)  # canonical init
        nn.init.normal_(self.down.weight, std=d_model**-0.5)
        nn.init.zeros_(self.up.weight)  # ksdraft doctrine: identity-block start

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, num_charts={self.num_charts}, "
                f"subspace_rank={self.subspace_rank}")

    def warp_scale(self) -> torch.Tensor:
        # softplus bound per the stabilized export; matches KS3Int8 packing.
        return F.softplus(self.raw_scale.float()) + 1e-4

    def warp_shift(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.raw_shift.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = self.down(x)
        geometry_dtype = (
            torch.float64 if coordinates.dtype == torch.float64 else torch.float32)
        h = coordinates.to(geometry_dtype).clamp(-65504.0, 65504.0)
        h = h.reshape(*x.shape[:-1], self.num_charts, self.subspace_rank)

        t = 4.0 * torch.tanh(h.mean(-1, keepdim=True) / 4.0)
        z = torch.tanh(self.warp_scale()[:, None] * (t + self.warp_shift()[:, None]))

        coeffs = self.coefficients.to(geometry_dtype) \
            .reshape(self.num_charts, self.subspace_rank, 6)
        spine, warped = spine_horner(t, z, coeffs)
        features = F.relu(h + spine + warped).square()
        return self.up(features.flatten(-2).to(self.up.weight.dtype))
