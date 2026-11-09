"""DraftBlock assembly (k-splanifolds RWKV-8-style hybrid shape):

    h   = x + RWKV7TimeMix(LN_rwkv(x)) + Rosa(LN_rosa(x))   # separate norms, summed
    h', p = AdjacentInjection(h)                            # DSpine depth conditioning
    y   = h' + KS3(LN_ffn(h'))

Init doctrine: every additive update starts at zero (rwkv output, rosa out
proj, ks3 up proj, injection up), so a fresh block is exactly identity.
"""

from __future__ import annotations

import torch
from torch import nn

from .ks3 import KS3
from .rosa import RosaMixer
from .rwkv7 import RWKV7TimeMix
from .spine import AdjacentInjection


class DraftBlock(nn.Module):
    def __init__(self, config, layer: int):
        super().__init__()
        d = config.d_model
        self.ln_rwkv = nn.LayerNorm(d)
        self.ln_rosa = nn.LayerNorm(d)
        self.time_mix = RWKV7TimeMix(d, layer, config.layers, config.head_size)
        self.rosa = RosaMixer(d, config.rosa_streams, config.rosa_bits)
        self.injection = AdjacentInjection(d)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = KS3(d, config.ks3_charts or None)

    def forward(
        self,
        x: torch.Tensor,
        first_value: torch.Tensor | None,
        prev_layer_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn, first_value = self.time_mix(self.ln_rwkv(x), first_value)
        h = x + attn + self.rosa(self.ln_rosa(x))
        h, feature = self.injection(h, prev_layer_feature)
        y = h + self.ffn(self.ln_ffn(h))
        return y, first_value, feature
