"""KSDraftModel — the assembled draft (reference tier).

Structure: tied embedding -> input norm -> DraftBlocks (Rosa + RWKV-7 delta
rule + DSpine injection + KS3) -> output norm -> tied head. The first block's
`first_value` seeds the first-layer value residual carried through the stack
(k-splanifolds Model behavior).

Dual-source conditioning seam (DFlash/H-Spec): optional projected teacher
states are added to the residual stream at the input. `teacher_dim` is the
flattened size of the fused multi-layer teacher states (n_taps * tap width);
zero disables the seam (tests / CE warm-up). The projection is zero-init.

dtype tiers (Section 12): construct in fp32, `.to(torch.bfloat16)` / fp32 for
the reference tiers; the int8 tier is the kernels package.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import KSDraftConfig
from .block import DraftBlock


class KSDraftModel(nn.Module):
    def __init__(self, config: KSDraftConfig | None = None, teacher_dim: int = 0,
                 device: torch.device | str | None = None, skip_init: bool = False):
        super().__init__()
        cfg = config or KSDraftConfig()
        self.config = cfg
        self.teacher_dim = teacher_dim

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList(
            [DraftBlock(cfg, layer) for layer in range(cfg.layers)])
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if teacher_dim:
            self.teacher_proj = nn.Linear(teacher_dim, cfg.d_model, bias=False)
        else:
            self.teacher_proj = None

        if device is not None:
            self.to(device)

        # Tie AFTER any device move: torch 2.10's .to() replaces Parameter
        # objects on conversion, which would silently break the tie.
        self.head.weight = self.embedding.weight

        if not skip_init:
            nn.init.uniform_(self.embedding.weight, -1e-4, 1e-4)
            if self.teacher_proj is not None:
                nn.init.zeros_(self.teacher_proj.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        teacher_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """tokens (B, T) -> (logits (B, T, V), per-layer transfer features).

        teacher_states: (B, T, teacher_dim) fused multi-layer target hidden
        states (already flattened across taps) for dual-source conditioning.
        """
        x = self.input_norm(self.embedding(tokens))
        if teacher_states is not None:
            if self.teacher_proj is None:
                raise ValueError("teacher_states passed but teacher_dim=0 at construction")
            # the projection owns the dtype boundary (signals arrive fp16)
            x = x + self.teacher_proj(
                teacher_states.to(self.teacher_proj.weight.dtype))

        prev_feature = None
        features = []
        for block in self.blocks:
            x, feature = block(x, prev_feature)
            prev_feature = feature
            features.append(feature)
        return self.head(self.output_norm(x)), features

    def param_counts(self) -> dict:
        core = self.config.layers * self.config.layer_params() + self.config.embedding_params()
        extras = sum(p.numel() for p in self.parameters()) - core
        return {"core": core, "extras": extras,
                "total": core + extras}
