"""Teacher clients (RESEARCH.md Sections 10 and 12A).

The teacher client is the toolkit's target-adapter seam. Three backends:

- ``FixtureTeacher``: deterministic synthetic signals behind the real
  interface — the teacher-free dev-box path (no model download, no GPU).
- ``HFTeacher``: any HF target behind ``from_pretrained``, frozen, with
  hidden-state taps captured by forward hooks; the generic fallback.
- The in-process PTQR INT8 client (int8-vllm loader) arrives with the MI100
  integration; it implements this same interface.

Signals captured per position (Section 11 Stage 1): multi-layer hidden
states at the configured taps + top-k logits (ids and FP16 probs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class TeacherSignals:
    states: torch.Tensor | None     # (B, T, n_taps, d_tap) fp16
    topk_ids: torch.Tensor          # (B, T, k) int64
    topk_probs: torch.Tensor        # (B, T, k) fp16

    def to(self, device, dtype=torch.float16):
        return TeacherSignals(
            self.states.to(device=device, dtype=dtype) if self.states is not None else None,
            self.topk_ids.to(device),
            self.topk_probs.to(device=device, dtype=dtype),
        )


class TeacherClient:
    """Interface every backend implements. Token ids in, signals out."""

    vocab_size: int
    n_taps: int
    topk: int

    def signals(self, tokens: torch.Tensor) -> TeacherSignals:
        raise NotImplementedError


class FixtureTeacher(TeacherClient):
    """Deterministic pseudo-teacher: seeded hash of (position, token, tap).

    Same tokens -> same signals across processes and runs, so training with
    the fixture is reproducible and exercises the full dual-source +
    distillation path without a target model. The 'top-k distribution' is a
    softmax over hashed logits — a valid probability simplex, deliberately
    uncorrelated with language (losses decrease by fitting the conditioning,
    not by the fixture being a real teacher)."""

    def __init__(self, vocab_size: int, d_tap: int = 64, n_taps: int = 6,
                 topk: int = 8, seed: int = 0):
        self.vocab_size = vocab_size
        self.d_tap = d_tap
        self.n_taps = n_taps
        self.topk = topk
        self.seed = seed

    def _hash(self, tokens: torch.Tensor, salt: int) -> torch.Tensor:
        g = torch.Generator(device=tokens.device)
        B, T = tokens.shape
        out = torch.empty(B, T, self.d_tap, device=tokens.device)
        for b in range(B):
            for t in range(T):
                g.manual_seed(int((int(tokens[b, t]) * 1000003 + t * 31 + b * 7919
                                   + salt * 104729 + self.seed) % (2**31)))
                out[b, t] = torch.randn(self.d_tap, generator=g,
                                        device=tokens.device)
        return out

    def signals(self, tokens: torch.Tensor) -> TeacherSignals:
        B, T = tokens.shape
        states = torch.stack(
            [self._hash(tokens, tap) for tap in range(self.n_taps)], dim=2).to(torch.float16)
        g = torch.Generator().manual_seed(self.seed + 1)
        proj = torch.randn(8, self.vocab_size, generator=g) * 0.05
        pos = self._hash8(tokens, 999)
        logits = pos.to(torch.float32) @ proj.to(tokens.device) \
            + 0.01 * tokens.to(torch.float32).unsqueeze(-1)
        probs = torch.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(self.topk, dim=-1)
        return TeacherSignals(states, topk_i, topk_p.to(torch.float16))

    def _hash8(self, tokens: torch.Tensor, salt: int) -> torch.Tensor:
        """(B, T, 8) hash used to synthesize per-position vocab logits."""
        g = torch.Generator(device=tokens.device)
        B, T = tokens.shape
        out = torch.empty(B, T, 8, device=tokens.device)
        for b in range(B):
            for t in range(T):
                g.manual_seed(int((int(tokens[b, t]) * 1000003 + t * 31 + b * 7919
                                   + salt * 104729 + self.seed) % (2**31)))
                out[b, t] = torch.randn(8, generator=g, device=tokens.device)
        return out


class HFTeacher(TeacherClient):
    """Frozen HF target with hidden-state taps (the universal fallback)."""

    def __init__(self, model_name: str, taps: Sequence[int], topk: int = 32,
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.taps = list(taps)
        self.topk = topk
        self.device = device
        self.vocab_size = self.model.config.vocab_size

    @property
    def n_taps(self) -> int:
        return len(self.taps)

    def signals(self, tokens: torch.Tensor) -> TeacherSignals:
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(idx):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h.detach()
            return hook

        try:
            layers = self.model.model.layers
        except AttributeError:
            layers = self.model.transformer.h
        for idx in self.taps:
            hooks.append(layers[idx].register_forward_hook(make_hook(idx)))
        try:
            with torch.no_grad():
                logits = self.model(tokens.to(self.device)).logits.float()
        finally:
            for h in hooks:
                h.remove()
        states = torch.stack([captured[i] for i in self.taps], dim=2).to(torch.float16)
        probs = torch.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(self.topk, dim=-1)
        return TeacherSignals(states, topk_i, topk_p.to(torch.float16))
