"""DFlash2-style pairwise path selector for the parallel block head.

The block head's offsets are independent unary distributions. DFlash2 keeps
the top-K candidates per position and scores every adjacent pair:

    S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>

where A/B are predecessor/successor codebooks and H is a context gate from
the position's hidden state. A greedy walk from the anchor token picks one
coherent path; `U` remains the unary draft logit so the path never leaves the
candidate set.

Low-rank implementation (~1.2M params): A/B are linear projections of the
tied target embedding rows, H projects the block hidden state; rank 256. The
codebook projections are zero-initialized, so the pair term is exactly 0 at
init and the selector reduces to the unary argmax until trained.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BlockSelector(nn.Module):
    def __init__(self, d_model: int, rank: int = 256):
        super().__init__()
        self.rank = rank
        self.a_proj = nn.Linear(d_model, rank, bias=False)
        self.b_proj = nn.Linear(d_model, rank, bias=False)
        self.ctx_proj = nn.Linear(d_model, rank, bias=False)
        for m in (self.a_proj, self.b_proj, self.ctx_proj):
            nn.init.zeros_(m.weight)      # pair term = 0 at init

    def score_candidates(self, h: torch.Tensor, unary: torch.Tensor,
                         cand_emb: torch.Tensor,
                         prev_emb: torch.Tensor) -> torch.Tensor:
        """Pair-wise scores for candidates whose embeddings are given.

        h/prev_emb: [N, d]; cand_emb: [N, K, d]; unary: [N, K].
        """
        a = self.a_proj(prev_emb)                 # [N, r]
        g = self.ctx_proj(h)                      # [N, r]
        b = self.b_proj(cand_emb)                 # [N, K, r]
        pair = ((a * g).unsqueeze(1) * b).sum(-1)
        return unary + pair
