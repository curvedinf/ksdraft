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
from .blockconv import BlockConv


class KSDraftModel(nn.Module):
    def __init__(self, config: KSDraftConfig | None = None, teacher_dim: int = 0,
                 device: torch.device | str | None = None, skip_init: bool = False,
                 embedding_init: torch.Tensor | None = None):
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
        # Parallel block drafting: N learned offsets on the output residual,
        # one tied-head projection per offset. One trunk forward then yields
        # N future-token distributions from the anchor position (Medusa-style
        # head; the alternative is N autoregressive forwards).
        self.block_delta = nn.Parameter(
            torch.zeros(cfg.block_len, cfg.d_model))
        # DFlash2-style intra-block conditioning on the offset states; identity
        # at init so existing checkpoints and exports are unchanged.
        self.block_conv = BlockConv(cfg.d_model)

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
        if embedding_init is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_init.to(
                    self.embedding.weight.dtype))

    def trunk(self, tokens: torch.Tensor,
              teacher_states: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Hidden states after the output norm, plus per-layer features."""
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
        return self.output_norm(x), features

    def forward(
        self,
        tokens: torch.Tensor,
        teacher_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """tokens (B, T) -> (logits (B, T, V), per-layer transfer features)."""
        h, features = self.trunk(tokens, teacher_states)
        return self.head(h), features

    def forward_block_hidden(self, tokens: torch.Tensor, n: int,
                             teacher_states: torch.Tensor | None = None
                             ) -> tuple[torch.Tensor, list[torch.Tensor],
                                        torch.Tensor]:
        """Trunk pass returning (hidden, features, block hidden states).

        Training projects these through the chunked head losses so the full
        vocab logits are never materialized (see kernels/fused_train.py).
        """
        h, features = self.trunk(tokens, teacher_states)
        anchor = h[:, -1]
        hs = self.block_conv(anchor.unsqueeze(1)
                             + self.block_delta[:n].unsqueeze(0))
        return h, features, hs

    def forward_block(self, tokens: torch.Tensor, n: int,
                      teacher_states: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """One trunk pass returning (logits, features, block logits).

        Training uses this so the block loss does not pay a second trunk
        forward (memory and time); serving uses block_logits().
        """
        h, features, hs = self.forward_block_hidden(tokens, n, teacher_states)
        return self.head(h), features, hs @ self.head.weight.t()

    def block_logits(self, tokens: torch.Tensor, n: int,
                     teacher_states: torch.Tensor | None = None
                     ) -> torch.Tensor:
        """One forward -> logits for the next ``n`` tokens after the anchor.

        Returns (B, n, V): offset k is the distribution for token anchor+k+1.
        Uses the last input position's hidden state plus the learned offset.
        """
        h, _ = self.trunk(tokens, teacher_states)
        anchor = h[:, -1]
        hs = self.block_conv(anchor.unsqueeze(1)
                             + self.block_delta[:n].unsqueeze(0))
        return hs @ self.head.weight.t()

    def param_counts(self) -> dict:
        core = self.config.layers * self.config.layer_params() + self.config.embedding_params()
        extras = sum(p.numel() for p in self.parameters()) - core
        return {"core": core, "extras": extras,
                "total": core + extras}
