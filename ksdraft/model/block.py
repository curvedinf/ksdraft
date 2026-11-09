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
    def __init__(self, config, layer: int):
        super().__init__()
        d = config.d_model
        self.ln_attn = nn.LayerNorm(d)
        self.rosa = RosaMixer(d, config.rosa_streams, config.rosa_bits)
        self.injection = AdjacentInjection(d)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = KS3(d, config.ks3_charts or None)

    def forward(
        self,
        x: torch.Tensor,
        prev_layer_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = x + self.rosa(self.ln_attn(x))
        h, feature = self.injection(h, prev_layer_feature)
        y = h + self.ffn(self.ln_ffn(h))
        return y, feature
