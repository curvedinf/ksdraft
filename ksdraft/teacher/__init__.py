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


def fixture_kv_stream(tokens: torch.Tensor, layers: int, kv_dim: int,
                      base_salt: int, seed: int = 0) -> torch.Tensor:
    """Deterministic synthetic per-layer KV stream, (B, layers, T, kv_dim).

    Same hash recipe as ``FixtureTeacher._hash`` (seeded per position), so
    fixture KV is reproducible across processes and runs.
    """
    B, T = tokens.shape
    out = torch.empty(B, layers, T, kv_dim, device=tokens.device)
    g = torch.Generator(device=tokens.device)
    for b in range(B):
        for t in range(T):
            for l in range(layers):
                g.manual_seed(int((int(tokens[b, t]) * 1000003 + t * 31 + b * 7919
                                   + (base_salt + l) * 104729 + seed) % (2**31)))
                out[b, l, t] = torch.randn(kv_dim, generator=g,
                                           device=tokens.device)
    return out


def blocks_int8_kv(stream: torch.Tensor,
                   group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Round a KV stream onto the deployed int8_block_g128 grid: codes+scales.

    Returns (codes int8 (..., Dp) trimmed to D, scales fp16 (..., Dp/group))
    with the fork cache-write kernel's exact semantics (see emulate_int8_kv).
    This is the int8 transport form of the KV seam: nothing here is ever
    materialized as a dequantized fp16 tensor.
    """
    shape = stream.shape
    D = shape[-1]
    dpad = (-D) % group
    x = stream.detach().float().reshape(-1, D)
    if dpad:
        x = torch.nn.functional.pad(x, (0, dpad))
    blocks = x.view(-1, (D + dpad) // group, group)
    amax = blocks.abs().amax(dim=-1)
    scale = (amax / 127.0).clamp_min(1e-6)
    scale = scale.to(torch.float16).to(torch.float32)
    q = blocks / scale.unsqueeze(-1)
    q = torch.where(q >= 0, q + 0.5, q - 0.5).trunc()   # half away from zero
    q = q.clamp_(-128.0, 127.0)
    codes = q.reshape(-1, D + dpad)[:, :D].to(torch.int8).view(shape)
    scales = scale.to(torch.float16).view(*shape[:-1], (D + dpad) // group)
    return codes, scales


def emulate_int8_kv(stream: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Round a KV stream onto the deployed int8_block_g128 grid.

    The serving stack caches target KV as int8_block_g128, so the draft's
    conditioning signal at serving is the dequantized grid, not clean fp16.
    Training through this pass keeps the train/serve KV distributions
    identical (the PTQR exact-tier-student doctrine applied to the reuse
    seam).

    Matches the fork's cache-write kernel exactly
    (vllm/v1/attention/ops/triton_reshape_and_cache_flash.py):
    - per-group scale = amax/127 floored at 1e-6, rounded via fp16 and used
      in that rounded form for encoding;
    - codes round HALF-AWAY-FROM-ZERO (not banker's rounding);
    - codes clamp to [-128, 127] (the asymmetric int8 range).
    Grouping runs along the flat channel dim: for Inkling's 8 KV heads of
    128 the group boundaries coincide with the kernel's per-head vectors.
    """
    codes, scales = blocks_int8_kv(stream, group)
    return dequant_kv(codes, scales, group)


def dequant_kv(codes: torch.Tensor, scales: torch.Tensor,
               group: int = 128) -> torch.Tensor:
    """Inverse of blocks_int8_kv (exact on the codes it produced)."""
    shape = codes.shape
    D = shape[-1]
    dpad = (-D) % group
    x = codes.reshape(-1, D).float()
    if dpad:
        x = torch.nn.functional.pad(x, (0, dpad))
    g = x.view(-1, (D + dpad) // group, group)
    s = scales.float().reshape(-1, (D + dpad) // group, 1)
    dq = (g * s).reshape(-1, D + dpad)[:, :D].view(shape)
    return dq


def causal_kv_windows(stream: torch.Tensor, window: int) -> torch.Tensor:
    """(B, L, T, D) stream -> (B, L, T, W, D) provider-aligned causal windows.

    Query t's window covers stream positions [t - W + 1, t] clamped at 0:
    causal by construction, the contract TargetKVAttention consumes.
    """
    B, L, T, D = stream.shape
    W = min(window, T)
    pos = (torch.arange(T, device=stream.device)[:, None]
           + torch.arange(W, device=stream.device)[None, :] - (W - 1)
           ).clamp_min(0)
    return stream[:, :, pos]


@dataclass
class TeacherSignals:
    states: torch.Tensor | None     # (B, T, n_taps, d_tap) fp16
    topk_ids: torch.Tensor          # (B, T, k) int64
    topk_probs: torch.Tensor        # (B, T, k) fp16
    # DFly2-style target-KV reuse: (k, v), each (B, layers, T, W, kv_dim)
    # fp16 — one provider-aligned causal window tap per draft layer.
    kv: tuple[torch.Tensor, torch.Tensor] | None = None

    def to(self, device, dtype=torch.float16):
        return TeacherSignals(
            self.states.to(device=device, dtype=dtype) if self.states is not None else None,
            self.topk_ids.to(device),
            self.topk_probs.to(device=device, dtype=dtype),
            kv=(tuple(t.to(device=device, dtype=dtype) for t in self.kv)
                if self.kv is not None else None),
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
                 topk: int = 8, seed: int = 0, kv_dim: int = 0,
                 kv_window: int = 16, kv_taps: int | None = None,
                 kv_quant_group: int = 128):
        self.vocab_size = vocab_size
        self.d_tap = d_tap
        self.n_taps = n_taps
        self.topk = topk
        self.seed = seed
        self.kv_dim = kv_dim
        self.kv_window = kv_window
        self.kv_taps = kv_taps if kv_taps is not None else n_taps
        self.kv_quant_group = kv_quant_group

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
        kv = None
        if self.kv_dim:
            k_stream = fixture_kv_stream(tokens, self.kv_taps, self.kv_dim,
                                         9_000, self.seed)
            v_stream = fixture_kv_stream(tokens, self.kv_taps, self.kv_dim,
                                         9_100, self.seed)
            if self.kv_quant_group:
                k_stream = emulate_int8_kv(k_stream, self.kv_quant_group)
                v_stream = emulate_int8_kv(v_stream, self.kv_quant_group)
            kv = (causal_kv_windows(k_stream, self.kv_window).to(torch.float16),
                  causal_kv_windows(v_stream, self.kv_window).to(torch.float16))
        return TeacherSignals(states, topk_i, topk_p.to(torch.float16), kv)

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


class KVFixture(TeacherClient):
    """Deterministic synthetic target KV only, behind the teacher seam.

    The teacherless reuse stages (warm-up, smoke tests) need a KV source
    even with no target model; this provider supplies the same hash recipe
    as ``FixtureTeacher``'s kv, without any logprob/states work. The topk
    fields are placeholders (never consumed on this path).
    """

    def __init__(self, layers: int, kv_dim: int, kv_window: int = 16,
                 seed: int = 0, kv_quant_group: int = 128):
        self.layers = int(layers)
        self.kv_dim = int(kv_dim)
        self.kv_window = int(kv_window)
        self.seed = seed
        self.kv_quant_group = kv_quant_group
        self.vocab_size = 0
        self.n_taps = self.layers
        self.topk = 1

    def signals(self, tokens: torch.Tensor) -> TeacherSignals:
        B, T = tokens.shape
        k_stream = fixture_kv_stream(tokens, self.layers, self.kv_dim,
                                     9_000, self.seed)
        v_stream = fixture_kv_stream(tokens, self.layers, self.kv_dim,
                                     9_100, self.seed)
        if self.kv_quant_group:
            k_stream = emulate_int8_kv(k_stream, self.kv_quant_group)
            v_stream = emulate_int8_kv(v_stream, self.kv_quant_group)
        kv = (causal_kv_windows(k_stream, self.kv_window).to(torch.float16),
              causal_kv_windows(v_stream, self.kv_window).to(torch.float16))
        return TeacherSignals(
            states=None,
            topk_ids=torch.zeros(B, T, 1, dtype=torch.int64),
            topk_probs=torch.ones(B, T, 1, dtype=torch.float16),
            kv=kv)


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


from .mined import MinedSignals  # noqa: E402,F401  (re-exported provider)
