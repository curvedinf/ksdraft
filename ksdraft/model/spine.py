"""DSpine gated adjacent injection, implemented to the paper's equations.

Reference: arXiv 2609.36173 (Tencent), Sections 4.1-4.4.

At every layer, after the layer's update:

    z_l,t   = sqrt(r) * normalize(R_l u_l,t)          # predicted feature
    g_l,t   = sigmoid(G_q z_l,t + G_p z_l,t-1 + b)    # shared G_q, G_p, b
    m_l,t   = A z_l,t-1                               # shared A
    h_l,t   = u_l,t + RMS(u_l,t) * W_l (g_l,t * m_l,t)   # per-layer W_l

A, G_q, G_p, b are shared across layers; W_l is per layer. Position 0 (the
anchor) has no in-block predecessor and stays unchanged. G_p, W_l and b are
zero-initialized, so every injection starts as the identity map.

Predicted features live in the unified transfer space C (a whitened PCA of
the target LM head): a known token enters as sqrt(r) C_x, and the layer-wise
embedding loss aligns the post-injection readout f_l(h_l,t) with C_{y_t}.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpineCore(nn.Module):
    """Parameters shared by every layer's injection (paper: A, G_q, G_p, b)."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(rank, rank, bias=False)
        self.Gq = nn.Linear(rank, rank, bias=False)
        self.Gp = nn.Linear(rank, rank, bias=False)
        self.b = nn.Parameter(torch.zeros(rank))
        nn.init.zeros_(self.Gp.weight)     # paper: G_p zero, W_l zero
        # A is NOT zero-initialized in the paper; the identity at init comes
        # from W_l = 0. Keep PyTorch's default init so A is trainable from
        # the start rather than from a degenerate zero map.


class AdjacentInjection(nn.Module):
    def __init__(self, width: int, rank: int, core: SpineCore,
                 transfer_C: torch.Tensor | None = None,
                 R_init: torch.Tensor | None = None):
        super().__init__()
        self.width = width
        self.rank = rank
        # shared core is intentionally not registered as a submodule here
        object.__setattr__(self, "_core", core)
        self.R = nn.Linear(width, rank, bias=False)     # layer readout
        self.W = nn.Linear(rank, width, bias=False)     # layer write
        nn.init.zeros_(self.W.weight)                   # identity at init
        if R_init is not None:
            with torch.no_grad():
                self.R.weight.copy_(R_init)
        if transfer_C is not None:
            self.register_buffer("C", transfer_C, persistent=False)
        else:
            self.C = None

    def extra_repr(self) -> str:
        return f"width={self.width}, rank={self.rank}"

    def feature(self, h: torch.Tensor) -> torch.Tensor:
        return self.rank ** 0.5 * F.normalize(self.R(h), dim=-1)

    def block_write(self, u: torch.Tensor,
                    prev_tokens: torch.Tensor) -> torch.Tensor:
        """Last-write refinement over block offsets (DSpine 4.3).

        u [B, n, d] pre-write offset states; prev_tokens [B, n] holds the
        *decided* predecessor token at each offset (the anchor token at
        offset 0). A known token enters the transfer space directly as
        sqrt(r) C_s, which changes both gate and message.
        """
        z = self.feature(u)
        if self.C is None:
            return u
        tok = self.C[prev_tokens].to(u.dtype).float() * self.rank ** 0.5
        z_prev = F.normalize(tok, dim=-1) * self.rank ** 0.5
        core = self._core
        g = torch.sigmoid(core.Gq(z) + core.Gp(z_prev) + core.b)
        m = core.A(z_prev)
        rms = u.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return u + rms * self.W(g * m)

    def forward(self, u: torch.Tensor, prev_tokens: torch.Tensor | None = None,
                sub_prob: float = 0.0):
        """u: (B, T, d) -> (post-injection h, transfer readout e)."""
        z = self.feature(u)
        z_prev = torch.zeros_like(z)
        z_prev[:, 1:] = z[:, :-1]
        if (self.training and sub_prob > 0.0 and self.C is not None
                and prev_tokens is not None):
            # teacher forcing: replace shallow predecessor features with the
            # correct token's transfer embedding, then phase out (paper 4.5)
            tok = self.C[prev_tokens].to(z.dtype) * self.rank ** 0.5
            tok = F.normalize(tok, dim=-1) * self.rank ** 0.5
            mask = (torch.rand(z.shape[:2], device=z.device) < sub_prob)
            mask[:, 0] = False
            z_prev = torch.where(mask.unsqueeze(-1), tok, z_prev)
        core = self._core
        g = torch.sigmoid(core.Gq(z) + core.Gp(z_prev) + core.b)
        m = core.A(z_prev)
        rms = u.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        h = u + rms * self.W(g * m)
        e = self.feature(h)                       # post-injection readout
        return h, e
