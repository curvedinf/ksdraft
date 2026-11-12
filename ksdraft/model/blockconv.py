"""Intra-block conditioning for the parallel block head (DFlash2-style).

DFlash predicts every block position independently, which is its documented
weakness: neighbouring draft tokens can be individually plausible but
incoherent as a path. DFlash2 adds a two-tap dynamic depthwise convolution
inside the draft backbone; this module is the same mechanism applied at the
block head, where our parallel offsets actually live:

    out_k = (base_0 + dyn_0(h_k)) * h_k
          + (base_1 + dyn_1(h_k)) * h_{k-1}

`base = [1, 0]` and `dyn = 0` at init, so a fresh model is exactly the
previous head (checkpoints and exports stay equivalent until trained). The
dynamic component is shared per channel group, matching DFlash2's
"every 16 channels share one correction".
"""

from __future__ import annotations

import torch
from torch import nn


class BlockConv(nn.Module):
    def __init__(self, d_model: int, taps: int = 2, group_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.taps = taps
        self.group_size = max(1, int(group_size))
        self.base = nn.Parameter(torch.zeros(taps, d_model))
        with torch.no_grad():
            self.base[0].fill_(1.0)          # identity at init
        self.dynamic = nn.Linear(d_model, taps * d_model, bias=False)
        nn.init.zeros_(self.dynamic.weight)  # no correction at init

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, taps={self.taps}"

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        """hs: (B, n, d) offset hidden states, offset 0 = first draft token."""
        b, n, d = hs.shape
        kern = self.dynamic(hs).view(b, n, self.taps, d)
        if self.group_size > 1 and d % self.group_size == 0:
            g = self.group_size
            kern = (kern.view(b, n, self.taps, d // g, g)
                        .mean(dim=-1, keepdim=True)
                        .expand(b, n, self.taps, d // g, g)
                        .reshape(b, n, self.taps, d))
        out = (self.base[0] + kern[:, :, 0]) * hs
        for t in range(1, self.taps):
            shifted = torch.zeros_like(hs)
            shifted[:, t:] = hs[:, :-t]          # h_{k-t}, zero at k < t
            out = out + (self.base[t] + kern[:, :, t]) * shifted
        return out
