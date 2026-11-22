"""KSDraftModel — the assembled draft (reference tier).

Structure: embedding (tied head by default; untied DFly2 pattern via
tie_word_embeddings=False, head initialized from the dequantized W4 target
unembed) -> input norm -> blocks -> output norm -> head. Blocks are Rosa
mixers by default (kv_dim=0) or DFly2-style target-KV reuse attention
(kv_dim>0, model/reuse.py); every block carries DSpine injection + KS3.
The first block's `first_value` seeds the first-layer value residual
carried through the stack (k-splanifolds Model behavior).

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
from .reuse import ReuseBlock
from .selector import BlockSelector
from .spine import AdjacentInjection, SpineCore


class KSDraftModel(nn.Module):
    def __init__(self, config: KSDraftConfig | None = None, teacher_dim: int = 0,
                 device: torch.device | str | None = None, skip_init: bool = False,
                 embedding_init: torch.Tensor | None = None,
                 head_init: torch.Tensor | None = None,
                 transfer_space: dict | None = None):
        super().__init__()
        cfg = config or KSDraftConfig()
        self.config = cfg
        self.teacher_dim = teacher_dim

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)

        # DSpine unified transfer space (fixed whitened PCA of the target LM
        # head). Without a file, fall back to unit-norm embedding rows at the
        # model width so tests/dev boxes stay constructible.
        if transfer_space is not None:
            C = transfer_space.get("C")
            R_init = transfer_space.get("R_init")
            rank = int(transfer_space.get(
                "rank", C.shape[-1] if C is not None else cfg.d_model))
            # Inference rebuilds (export/load) carry the rank only; C is a
            # training-time table and is never used on the serving path.
            # The transfer table is needed by last-write refinement at
            # serving too, so it is a persistent fp16 buffer (77 MB).
            C = (C.half() if C is not None
                 else torch.zeros(cfg.vocab_size, rank, dtype=torch.float16))
        else:
            C = torch.nn.functional.normalize(
                self.embedding.weight.detach().float(), dim=-1).half()
            R_init = None
            rank = cfg.d_model
        self.register_buffer("transfer_C", C, persistent=True)
        self.spine_core = SpineCore(rank)
        block_cls = ReuseBlock if cfg.kv_dim else DraftBlock
        self.blocks = nn.ModuleList(
            [block_cls(cfg, layer, self.spine_core, rank, C, R_init)
             for layer in range(cfg.layers)])
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
        # DFlash2 pairwise path selector over the block head's unary top-K.
        self.block_selector = BlockSelector(cfg.d_model, rank=256)
        # DSpine last-write refinement at the block head: the same injection
        # core, re-run with the decided predecessor token's transfer row.
        self.head_write = AdjacentInjection(cfg.d_model, rank, self.spine_core,
                                            C, R_init)

        if teacher_dim:
            self.teacher_proj = nn.Linear(teacher_dim, cfg.d_model, bias=False)
        else:
            self.teacher_proj = None

        if device is not None:
            self.to(device)

        # Tie AFTER any device move: torch 2.10's .to() replaces Parameter
        # objects on conversion, which would silently break the tie. An
        # untied draft (DFly2 pattern) keeps the head as its own table —
        # typically the dequantized W4 target unembed, frozen.
        if cfg.tie_word_embeddings:
            self.head.weight = self.embedding.weight

        if not skip_init:
            nn.init.uniform_(self.embedding.weight, -1e-4, 1e-4)
            if self.teacher_proj is not None:
                nn.init.zeros_(self.teacher_proj.weight)
        if embedding_init is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_init.to(
                    self.embedding.weight.dtype))
        if head_init is not None:
            if cfg.tie_word_embeddings:
                raise ValueError(
                    "head_init passed but tie_word_embeddings=True — untie "
                    "in the config to give the head its own table")
            with torch.no_grad():
                self.head.weight.copy_(head_init.to(self.head.weight.dtype))

    def trunk(self, tokens: torch.Tensor,
              teacher_states: torch.Tensor | None = None,
              target_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
              sub_prob: float = 0.0
              ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Hidden states after the output norm, plus per-layer readouts.

        Teacher-forced predecessor tokens feed the optional DSpine
        substitution (paper 4.5); position 0 has no in-block predecessor.
        ``target_kv`` is the reuse backbone's context: (k, v), each
        (B, layers, T, W, kv_dim), one provider-aligned causal window tap
        per draft layer.
        """
        x = self.input_norm(self.embedding(tokens))
        if teacher_states is not None:
            if self.teacher_proj is None:
                raise ValueError("teacher_states passed but teacher_dim=0 at construction")
            # the projection owns the dtype boundary (signals arrive fp16)
            x = x + self.teacher_proj(
                teacher_states.to(self.teacher_proj.weight.dtype))

        k_src = v_src = None
        if self.config.kv_dim:
            if target_kv is None:
                raise ValueError(
                    "kv_dim > 0 (reuse backbone) requires target_kv=(k, v) "
                    "at every forward — pass a KV provider through the trainer "
                    "or the serving seam")
            k_src, v_src = target_kv
            if k_src.shape[1] != len(self.blocks):
                raise ValueError(
                    f"target_kv carries {k_src.shape[1]} layer taps but the "
                    f"model has {len(self.blocks)} blocks")
        elif target_kv is not None:
            raise ValueError("target_kv passed to a Rosa-backbone model "
                             "(kv_dim=0)")

        prev_tokens = torch.zeros_like(tokens)
        prev_tokens[:, 1:] = tokens[:, :-1]
        features = []
        for i, block in enumerate(self.blocks):
            if k_src is not None:
                x, feature = block(x, prev_tokens, k_src[:, i], v_src[:, i],
                                   sub_prob)
            else:
                x, feature = block(x, prev_tokens, sub_prob)
            features.append(feature)
        return self.output_norm(x), features

    def forward(
        self,
        tokens: torch.Tensor,
        teacher_states: torch.Tensor | None = None,
        target_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """tokens (B, T) -> (logits (B, T, V), per-layer transfer features)."""
        h, features = self.trunk(tokens, teacher_states, target_kv)
        return self.head(h), features

    def forward_block_hidden(self, tokens: torch.Tensor, n: int,
                             teacher_states: torch.Tensor | None = None,
                             target_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                             sub_prob: float = 0.0
                             ) -> tuple[torch.Tensor, list[torch.Tensor],
                                        torch.Tensor]:
        """Trunk pass returning (hidden, features, block hidden states).

        Training projects these through the chunked head losses so the full
        vocab logits are never materialized (see kernels/fused_train.py).
        """
        h, features = self.trunk(tokens, teacher_states, target_kv, sub_prob)
        anchor = h[:, -1]
        hs = self.block_conv(anchor.unsqueeze(1)
                             + self.block_delta[:n].unsqueeze(0))
        return h, features, hs

    def forward_block(self, tokens: torch.Tensor, n: int,
                      teacher_states: torch.Tensor | None = None,
                      target_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                      ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """One trunk pass returning (logits, features, block logits).

        Training uses this so the block loss does not pay a second trunk
        forward (memory and time); serving uses block_logits().
        """
        h, features, hs = self.forward_block_hidden(tokens, n, teacher_states,
                                                    target_kv)
        return self.head(h), features, hs @ self.head.weight.t()

    def selector_loss(self, hs: torch.Tensor, tokens: torch.Tensor,
                      anchor: int, bn: int, topk: int = 16) -> torch.Tensor:
        """DFlash2 selector CE over each offset's unary top-K candidates.

        The predecessor is teacher-forced: offset j scores candidate b given
        token anchor+j; the label is token anchor+1+j. The target replaces
        the weakest unary candidate when it is outside the top-K (Speculators
        behaviour) so the K-way CE is always defined.
        """
        from ..kernels.fused_train import linear_topk

        b, n, d = hs.shape
        flat = hs.reshape(-1, d)
        vals, ids = linear_topk(flat, self.head.weight, topk)
        n_rows = b * n
        pred = torch.stack([tokens[:, anchor + j] for j in range(bn)],
                           1).reshape(-1)
        tgt = torch.stack([tokens[:, anchor + 1 + j] for j in range(bn)],
                          1).reshape(-1)
        emb = self.embedding.weight
        # replace the weakest candidate when the label is absent
        present = (ids == tgt[:, None]).any(-1)
        if not bool(present.all()):
            row = torch.arange(n_rows, device=hs.device)
            w_tgt = emb[tgt]
            u_tgt = (flat * w_tgt).sum(-1)
            ids[row, -1] = tgt
            vals[row, -1] = u_tgt
        cand_emb = emb[ids]
        prev_emb = emb[pred]
        scores = self.block_selector.score_candidates(
            flat, vals, cand_emb, prev_emb)
        target_idx = (ids == tgt[:, None]).float().argmax(-1)
        return torch.nn.functional.cross_entropy(scores, target_idx)

    def lwr_loss(self, hs: torch.Tensor, tokens: torch.Tensor,
                 anchor: int, bn: int, topk: int = 16) -> torch.Tensor:
        """DSpine last-write refinement loss at the block head.

        The offsets are re-written with the *correct* predecessor token
        (teacher forcing), then scored only over each offset's unary top-K
        candidates; the target replaces the weakest candidate when absent.
        """
        from ..kernels.fused_train import linear_topk

        b, n, d = hs.shape
        flat = hs.reshape(-1, d)
        vals, ids = linear_topk(flat, self.head.weight, topk)
        prev = torch.stack([tokens[:, anchor + j] for j in range(bn)],
                           1).reshape(-1)
        tgt = torch.stack([tokens[:, anchor + 1 + j] for j in range(bn)],
                          1).reshape(-1)
        h = self.head_write.block_write(hs, prev.view(b, n)).reshape(-1, d)
        weight = self.head.weight
        rows = weight[ids]                       # [N, K, d]
        scores = (h.unsqueeze(1) * rows).sum(-1)
        present = (ids == tgt[:, None]).any(-1)
        label_ids = ids
        if not bool(present.all()):
            scores = scores.clone()
            label_ids = ids.clone()              # keep `ids` intact for autograd
            r = torch.arange(scores.shape[0], device=hs.device)
            scores[r, -1] = (h * weight[tgt]).sum(-1)
            label_ids[r, -1] = tgt
        target_idx = (label_ids == tgt[:, None]).float().argmax(-1)
        return torch.nn.functional.cross_entropy(scores, target_idx)

    @torch.no_grad()
    def propose_block_tokens(self, tokens: torch.Tensor, n: int,
                             topk: int = 16, return_conf: bool = False,
                             refine_iters: int = 0,
                             target_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
        """Greedy selector walk over the block candidates (serving path).

        Starts from the last verified token and conditions each position on
        the token selected at the previous position.
        """
        from ..kernels.fused_train import linear_topk

        self.eval()
        _h, _f, hs = self.forward_block_hidden(tokens, n, target_kv=target_kv)
        b, nn_, d = hs.shape
        vals, ids = linear_topk(hs.reshape(-1, d), self.head.weight, topk)
        vals = vals.view(b, nn_, topk)
        ids = ids.view(b, nn_, topk)
        weight = self.head.weight
        anchor_tok = tokens[:, -1]

        def walk(forced):
            prev = anchor_tok
            out, conf = [], []
            for j in range(nn_):
                # last-write refinement with the decided predecessor token
                h = self.head_write.block_write(hs[:, j:j + 1],
                                                prev[:, None])[:, 0]
                scores = (h.unsqueeze(1) * weight[ids[:, j]]).sum(-1)
                pick = scores.argmax(-1)
                tok = ids[:, j].gather(1, pick[:, None]).squeeze(1)
                out.append(tok)
                if return_conf:
                    p = torch.softmax(scores.float(), dim=-1)
                    conf.append(p.gather(1, pick[:, None]).squeeze(1))
                prev = tok if forced is None else forced[:, j]
            return torch.stack(out, dim=1), (
                torch.stack(conf, dim=1) if return_conf else None)

        toks, conf = walk(None)
        # JetSpec++ intra-block self refinement: re-run the walk with the
        # previously selected chain as the predecessor path; fixed point.
        for _ in range(int(refine_iters)):
            new, new_conf = walk(toks)
            same = torch.equal(new, toks)
            toks, conf = new, new_conf
            if same:
                break
        if return_conf:
            return toks, conf
        return toks

    def block_logits(self, tokens: torch.Tensor, n: int,
                     teacher_states: torch.Tensor | None = None,
                     target_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                     ) -> torch.Tensor:
        """One forward -> logits for the next ``n`` tokens after the anchor.

        Returns (B, n, V): offset k is the distribution for token anchor+k+1.
        Uses the last input position's hidden state plus the learned offset.
        """
        h, _ = self.trunk(tokens, teacher_states, target_kv)
        anchor = h[:, -1]
        hs = self.block_conv(anchor.unsqueeze(1)
                             + self.block_delta[:n].unsqueeze(0))
        return hs @ self.head.weight.t()

    def param_counts(self) -> dict:
        core = self.config.layers * self.config.layer_params() + self.config.embedding_params()
        extras = sum(p.numel() for p in self.parameters()) - core
        return {"core": core, "extras": extras,
                "total": core + extras}
