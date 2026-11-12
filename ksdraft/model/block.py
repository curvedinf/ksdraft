"""DraftBlock assembly — the RWKV-8 block: Rosa mixer + DSpine injection + KS3.

Per the project decision the backbone is RWKV-8 proper: the Rosa hard-retrieval
mixer IS the time-mixing layer (no RWKV-7 delta-rule component). The block:

    h    = x + Rosa(LN_attn(x))
    h', p = AdjacentInjection(h)              # DSpine depth conditioning
    y    = h' + KS3(LN_ffn(h'))

Init doctrine: every additive update starts at zero (rosa out proj, injection
up, ks3 up), so a fresh block is exactly identity.
"""

from __future__ import annotations

import torch
from torch import nn

from .ks3 import KS3
from .rosa import RosaMixer
from .spine import AdjacentInjection


class DraftBlock(nn.Module):
    def __init__(self, config, layer: int, spine_core, rank: int,
                 transfer_C: torch.Tensor | None = None,
                 R_init: torch.Tensor | None = None):
        super().__init__()
        d = config.d_model
        self.ln_attn = nn.LayerNorm(d)
        self.rosa = RosaMixer(d, config.rosa_streams, config.rosa_bits)
        self.injection = AdjacentInjection(d, rank, spine_core, transfer_C,
                                           R_init)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = KS3(d, config.ks3_charts or None)

    def forward(
        self,
        x: torch.Tensor,
        prev_tokens: torch.Tensor | None = None,
        sub_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DSpine injects after the layer update (paper: after every layer),
        # so the FFN comes first and the message enters the post-layer state.
        h = x + self.rosa(self.ln_attn(x))
        y = h + self.ffn(self.ln_ffn(h))
        y, feature = self.injection(y, prev_tokens, sub_prob)
        return y, feature
