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
            if x.is_cuda:
                from ..kernels.rosa_credit_gpu import HardRosaGPU
                hard = HardRosaGPU.apply(q_full, k_full, v_full, self.payload)
                return self.out_proj(hard.to(x.dtype))
            from ..kernels.rosa_credit import engine_available, rosa_credit_forward
            if engine_available():
                hard = rosa_credit_forward(q_full, k_full, v_full, self.payload)
                return self.out_proj(hard.to(x.dtype))
        else:
            q_full = k_full = v_full = None
        qbits = (self.q_proj(q_in) > 0).to(torch.int8)   # (B, T, S)
        kbits = (self.k_proj(k_in) > 0).to(torch.int8)
        vbits = (self.v_proj(v_in) > 0).to(torch.int8)

        # per-stream match over (B, S) leading dims
        sign = _match_payload_sign(
            qbits.permute(0, 2, 1), kbits.permute(0, 2, 1), vbits.permute(0, 2, 1)
        )  # (B, S, T)
        out = self.payload[None, :, None] * sign         # (B, S, T) fp32
        return self.out_proj(out.permute(0, 2, 1).to(x.dtype))
