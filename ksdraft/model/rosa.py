"""RWKV-8 Rosa / QKV-ROSA hard-retrieval mixer — reference implementation.

Architecture from Bo Peng (BlinkDL) RWKV-v8 examples (samx_qkv_slow /
ROSA_QKV_B_1bit); integration shape per the k-splanifolds "RWKV-8-style RWKV7 +
ROSA hybrid" (curvedinf/k-splanifolds, experiments/rwkv7_ks2):

- `streams` independent 1-bit Q/K/V symbol streams.
- Three biased input projections d->s (one each for Q, K, V) applied to RWKV
  time-shifted inputs (three learned d-dim time-mix vectors, sigmoid lerp).
- Hard forward, per stream and position t: find the longest suffix of the
  query-bit history ending at t that equals a key-bit suffix ending at some
  strictly earlier position j; among maximal-length matches take the latest j
  ("endpoint"); read that endpoint's *successor* value bit v[j+1]. A matched
  one contributes +payload, a matched zero or no match contributes -payload
  (the pinned 1-bit hybrid contract).
- Payload scales per stream; biased output projection s->d.

Initialization doctrine (k-splanifolds): output weights/biases start at zero,
payloads at one, time mixes at zero — a fresh Rosa contributes nothing.

Reference limitations (the INT8/production tier removes them):
- The suffix match is computed with exact integer codes capped at
  `max_match_len = 62 // bits` (int64 safety). The C++ engine / Triton kernel
  use suffix automata with no length cap.
- bits are hard sign() decisions: gradients reach the payload scales and the
  output projection, but NOT the input projections/time mixes — that is the
  hard-symbol nature of Rosa and precisely why the counterfactual-credit
  backward exists (ksdraft.kernels, later milestone). This module is the
  correctness/parity reference, not the trainable path.
- Sequence-level forward only (no incremental decode state yet).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _match_payload_sign(
    qbits: torch.Tensor, kbits: torch.Tensor, vbits: torch.Tensor,
    max_match_len: int | None = None,
) -> torch.Tensor:
    """Longest-suffix-match payload signs.

    qbits/kbits/vbits: (..., T) int8 in {0, 1}. Returns (..., T) float in
    {+1, -1}: the successor-value sign at the matched endpoint, or -1 when no
    match exists. Vectorized over leading batch/stream dims.
    """
    *lead, t_len = qbits.shape
    device = qbits.device
    if max_match_len is None:
        max_match_len = max(1, min(62, t_len - 1)) if t_len > 1 else 0
    else:
        max_match_len = min(max_match_len, t_len - 1)

    sign = torch.full(lead + [t_len], -1.0, device=device, dtype=torch.float32)
    if max_match_len < 1:
        return sign

    # Rolling exact codes: code_L[t] = 2 * code_{L-1}[t-1] + bit[t]
    # (little-endian: most recent bit is the low bit). Lengths are searched
    # LONGEST FIRST: the first length with a match decides the position.
    qcode = torch.zeros(lead + [t_len], device=device, dtype=torch.int64)
    kcode = torch.zeros_like(qcode)
    codes: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _length in range(1, max_match_len + 1):
        qcode = torch.cat(
            [torch.zeros(lead + [1], device=device, dtype=torch.int64),
             2 * qcode[..., :-1]], dim=-1) + qbits.to(torch.int64)
        kcode = torch.cat(
            [torch.zeros(lead + [1], device=device, dtype=torch.int64),
             2 * kcode[..., :-1]], dim=-1) + kbits.to(torch.int64)
        codes.append((qcode, kcode))

    sign = torch.full(lead + [t_len], -1.0, device=device, dtype=torch.float32)
    decided = torch.zeros(lead + [t_len], device=device, dtype=torch.bool)
    idx = torch.arange(t_len, device=device)
    j_idx = idx.expand(lead + [t_len, t_len])

    for length in range(max_match_len, 0, -1):
        qcode, kcode = codes[length - 1]
        # match[t, j]: codes equal, j in [length-1, t-1]
        eq = qcode[..., :, None] == kcode[..., None, :]  # (..., T, T)
        valid = (idx[None, :] >= length - 1) & (idx[None, :] < idx[:, None])  # (T,T)
        cand = eq & valid & ~decided[..., :, None]
        has = cand.any(dim=-1)  # (..., T)
        take = has & ~decided
        if take.any():
            latest = torch.where(cand, j_idx, torch.full_like(j_idx, -1)).max(dim=-1).values
            succ = latest.clamp(min=0) + 1
            v_sign = torch.where(vbits.gather(-1, succ) > 0, 1.0, -1.0)
            sign = torch.where(take, v_sign, sign)
            decided = decided | take
        if bool(decided.all()):
            break

    return sign


class RosaMixer(nn.Module):
    def __init__(self, d_model: int, streams: int = 512, bits: int = 1):
        super().__init__()
        if bits != 1:
            raise NotImplementedError(
                "reference implements 1-bit streams (the decided configuration); "
                "n-bit packed symbols arrive with the production kernel"
            )
        self.d_model = d_model
        self.streams = streams
        self.bits = bits

        self.q_proj = nn.Linear(d_model, streams, bias=True)
        self.k_proj = nn.Linear(d_model, streams, bias=True)
        self.v_proj = nn.Linear(d_model, streams, bias=True)
        self.out_proj = nn.Linear(streams, d_model, bias=True)
        self.time_mix_q = nn.Parameter(torch.zeros(d_model))
        self.time_mix_k = nn.Parameter(torch.zeros(d_model))
        self.time_mix_v = nn.Parameter(torch.zeros(d_model))
        self.payload = nn.Parameter(torch.ones(streams))
        # Training-mode gradient path for the hard symbol core:
        #   "exact"  — batched counterfactual credits (HardRosaGPU; O(T^4),
        #              capped at T <= 63; the parity reference, used by gates
        #              and short fine-tunes)
        #   "engine" — external exact credit engine when available
        #   "ste"    — straight-through estimator: forward value is the exact
        #              hard match, backward is the identity on the
        #              projections (and therefore on the time mixes). This is
        #              the only mode fast enough for full training.
        self.credit_mode = "ste"

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, streams={self.streams}, bits={self.bits}"

    def _shifted(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prev = F.pad(x, (0, 0, 1, 0))[:, :-1, :]
        q_in = prev + (x - prev) * torch.sigmoid(self.time_mix_q)
        k_in = prev + (x - prev) * torch.sigmoid(self.time_mix_k)
        v_in = prev + (x - prev) * torch.sigmoid(self.time_mix_v)
        return q_in, k_in, v_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> (B, T, d).

        In training mode, when the exact-credit engine is available, the
        hard core runs through it so the symbol projections receive
        counterfactual-credit gradients (RESEARCH.md Section 7 pillar).
        Otherwise the torch reference path runs (payload/out_proj grads
        only — the documented hard-symbol gradient boundary)."""
        q_in, k_in, v_in = self._shifted(x)
        if self.training:
            q_full, k_full, v_full = self.q_proj(q_in), self.k_proj(k_in), self.v_proj(v_in)
            mode = getattr(self, "credit_mode", "ste")
            if mode == "exact":
                from ..kernels.rosa_credit_gpu import MAX_WINDOW, HardRosaGPU
                if q_full.shape[1] <= MAX_WINDOW:
                    hard = HardRosaGPU.apply(q_full, k_full, v_full, self.payload)
                    return self.out_proj(hard.to(x.dtype))
            if mode in ("exact", "engine"):
                from ..kernels.rosa_credit import engine_available, rosa_credit_forward
                if engine_available():
                    hard = rosa_credit_forward(q_full, k_full, v_full, self.payload)
                    return self.out_proj(hard.to(x.dtype))
            # STE: exact hard forward, identity gradient on the projections.
            # Adding (x - x.detach()) leaves the value unchanged (numerically
            # zero) while giving the projections/time mixes a gradient.
            with torch.no_grad():
                bits = ((q_full > 0).to(torch.int8).permute(0, 2, 1),
                        (k_full > 0).to(torch.int8).permute(0, 2, 1),
                        (v_full > 0).to(torch.int8).permute(0, 2, 1))
                sign = _match_payload_sign_scan(*bits)            # (B, S, T)
            hard = (self.payload[None, :, None] * sign).permute(0, 2, 1)
            hard = hard + (q_full - q_full.detach()) \
                        + (k_full - k_full.detach()) \
                        + (v_full - v_full.detach())
            return self.out_proj(hard.to(x.dtype))
        q_full = k_full = v_full = None
        with torch.no_grad():
            qbits = (self.q_proj(q_in) > 0).to(torch.int8)   # (B, T, S)
            kbits = (self.k_proj(k_in) > 0).to(torch.int8)
            vbits = (self.v_proj(v_in) > 0).to(torch.int8)

            # per-stream match over (B, S) leading dims
            sign = _match_payload_sign_scan(
                qbits.permute(0, 2, 1), kbits.permute(0, 2, 1), vbits.permute(0, 2, 1)
            )  # (B, S, T)
        out = self.payload[None, :, None] * sign         # (B, S, T) fp32
        return self.out_proj(out.permute(0, 2, 1).to(x.dtype))


def _match_payload_sign_fast(
    qbits: torch.Tensor, kbits: torch.Tensor, vbits: torch.Tensor,
    max_match_len: int | None = None,
) -> torch.Tensor:
    """Longest-suffix-match payload signs, O(T^2) DP instead of O(T^3) codes.

    ``lcs[t, j]`` = length of the longest common suffix of ``q[..t]`` and
    ``k[..j]``: ``lcs = (q[t] == k[j]) ? lcs[t-1, j-1] + 1 : 0``. For each
    position the longest length with any ``j < t`` decides, and among those
    the latest ``j`` provides the successor value bit. Same semantics as
    :func:`_match_payload_sign` (which is the rolling-int64 reference), with
    the same ``max_match_len`` cap for parity.
    """
    *lead, t_len = qbits.shape
    device = qbits.device
    if max_match_len is None:
        max_match_len = max(1, min(62, t_len - 1)) if t_len > 1 else 0
    else:
        max_match_len = min(max_match_len, max(t_len - 1, 0))

    sign = torch.full(lead + [t_len], -1.0, device=device, dtype=torch.float32)
    if max_match_len < 1:
        return sign

    q = qbits.float()
    k = kbits.float()
    v = vbits.float()
    prev = torch.zeros(lead + [t_len], device=device, dtype=torch.float32)
    for t in range(t_len):
        # lcs[t, j] for every j: (q[t] == k[j]) * (lcs[t-1, j-1] + 1)
        shifted = torch.cat(
            [torch.zeros(lead + [1], device=device), prev[..., :-1]], dim=-1)
        row = (q[..., t:t + 1] == k).float() * (shifted + 1.0)
        if max_match_len:
            row = row.clamp_max(float(max_match_len))
        eligible = row.clone()
        eligible[..., t:] = -1.0          # j must be strictly earlier than t
        best = eligible.max(dim=-1)
        m = best.values
        has = m > 0
        if bool(has.any()):
            latest = torch.where(
                eligible == m.unsqueeze(-1),
                torch.arange(t_len, device=device).expand_as(eligible),
                torch.full_like(eligible, -1)).max(dim=-1).values
            succ = (latest.clamp_min(0) + 1).clamp_max(t_len - 1).long()
            v_succ = torch.gather(v, -1, succ.unsqueeze(-1)).squeeze(-1)
            v_sign = torch.where(v_succ > 0, 1.0, -1.0)
            sign[..., t] = torch.where(has, v_sign, sign[..., t])
        prev = row
    return sign


def _match_payload_sign_scan(
    qbits: torch.Tensor, kbits: torch.Tensor, vbits: torch.Tensor,
    max_match_len: int | None = None,
) -> torch.Tensor:
    """Longest-suffix-match payload signs, fully vectorized in t.

    Along the anti-diagonal ``t - j = const`` the recurrence
    ``lcs[t, j] = E[t, j] ? lcs[t-1, j-1] + 1 : 0`` is a run length of the
    equality matrix. A run length is ``t - (last mismatch index)``, and the
    last-mismatch field is a prefix maximum along each diagonal, computed in
    ``log T`` doubling steps (shift the matrix down-right, take elementwise
    max). Everything runs as dense (T, T) tensor ops on ``lead + (T, T)``.
    """
    *lead, t_len = qbits.shape
    device = qbits.device
    if max_match_len is None:
        cap = max(1, min(62, t_len - 1)) if t_len > 1 else 0
    else:
        cap = min(max_match_len, max(t_len - 1, 0))
    sign = torch.full(lead + [t_len], -1.0, device=device, dtype=torch.float32)
    if cap < 1:
        return sign

    q = qbits.float()
    k = kbits.float()
    v = vbits.float()
    eq = (q[..., :, None] == k[..., None, :])          # (..., T, T): E[t, j]
    idx = torch.arange(t_len, device=device)
    # last mismatch position along the diagonal, as a t-index; -1 when none.
    last = torch.where(eq, torch.full_like(eq, -1.0, dtype=torch.float32),
                       idx[:, None].expand_as(eq).float())
    p = last
    s = 1
    while s < t_len:
        sh = torch.full_like(p, -1.0)
        if s < t_len:
            sh[..., s:, s:] = p[..., :-s, :-s]
        p = torch.maximum(p, sh)
        s <<= 1
    # match length at (t, j); clamp to the diagonal start and the code cap
    diag_start = (idx[:, None] - idx[None, :]).clamp_min(0)
    run = idx[:, None].float() - p
    run = torch.minimum(run, (idx[:, None] - diag_start + 1).float())
    run = run.clamp(0.0, float(cap))
    # eligible j < t; longest length decides, latest j breaks ties
    eligible = torch.where(idx[None, :] < idx[:, None], run,
                           torch.full_like(run, -1.0))  # (..., T, T)
    best, _ = eligible.max(dim=-1)                       # (..., T)
    has = best > 0
    latest = torch.where(
        eligible == best.unsqueeze(-1),
        idx.expand_as(eligible), torch.full_like(eligible, -1)).max(dim=-1).values
    succ = (latest.clamp_min(0) + 1).clamp_max(t_len - 1).long()
    v_succ = torch.gather(v, -1, succ)
    v_sign = torch.where(v_succ > 0, 1.0, -1.0)
    return torch.where(has, v_sign, sign)
