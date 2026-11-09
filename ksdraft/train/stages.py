"""Training stages 4-7 (RESEARCH.md Section 11) and PTQR dither annealing.

- Stage 4 (foresight): a lightweight low-rank adapter predicts the target's
  hidden state and the anchor token at block boundaries; the drafter stays
  frozen (EAGLE-4 freeze-then-head staging). Supervised by teacher signals.
- Stage 5 (refine): draft-forced self-refinement — continuations sampled
  from the drafter itself are scored against the teacher's distribution.
- Stage 6 (acceptance): Window Total Variation loss on the teacher's top-k
  support (the T=1.0 serving regime's operative objective), with an
  expected-accepted-length diagnostic for greedy.
- Stage 7 (ptqr): geometrically annealed rounding dither on the INT8
  quantization points (explore neighboring grid points early; bit-identical
  to serving at dither=0), then the fidelity gates: quantized top-1
  mismatch must equal the unquantized drafter's and acceptance must stay
  within +-1pt.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# ---------- dithered quantization (PTQR annealing) ----------

def quantize_last_dim_dithered(x: torch.Tensor, group: int, dither: float,
                               generator: torch.Generator | None = None):
    """GS quantization with rounding dither: uniform noise in
    [-dither, dither] code-steps added before rounding. dither=0 recovers
    the exact deterministic grid; dither=0.5 explores both neighbors."""
    from ..kernels.quant import (GROUP_SIZE, SCALE_DTYPE, SCALE_MAX, SCALE_MIN,
                                 _padded_length)
    K = x.shape[-1]
    Kp = _padded_length(K, group)
    if Kp != K:
        x = F.pad(x, (0, Kp - K))
    grouped = x.reshape(*x.shape[:-1], Kp // group, group)
    amax = grouped.abs().amax(dim=-1)
    scale = (amax.float() / 127.0).clamp(SCALE_MIN, SCALE_MAX)
    scale = scale.to(SCALE_DTYPE).to(torch.float32)
    codes = grouped / scale[..., None]
    if dither > 0:
        noise = (torch.rand(codes.shape, generator=generator,
                            device=codes.device, dtype=codes.dtype)
                 - 0.5) * (2.0 * dither)
        codes = codes + noise
    codes = torch.clamp(torch.round(codes), -127.0, 127.0)
    return codes.to(torch.int8).reshape(*x.shape[:-1], Kp)[..., :K].contiguous(), \
        scale.to(SCALE_DTYPE)


# ---------- losses ----------

def wtv_loss(logits: torch.Tensor, teacher_probs: torch.Tensor,
             teacher_topk_ids: torch.Tensor) -> torch.Tensor:
    """Window Total Variation on the teacher's (renormalized) top-k support:
    0.5 * (sum |p_model - p_teacher| on support + model mass off support)."""
    p_t = teacher_probs.float()
    p_t = p_t / p_t.sum(-1, keepdim=True).clamp(min=1e-9)
    p_m = torch.softmax(logits.float(), dim=-1)
    m_on = p_m.gather(-1, teacher_topk_ids)            # (B, T, k)
    on_diff = (m_on - p_t).abs().sum(-1)
    off_mass = (1.0 - m_on.sum(-1)).clamp(min=0.0)
    return 0.5 * (on_diff + off_mass).mean()


def expected_accepted_length(logits: torch.Tensor, tokens: torch.Tensor,
                             teacher_probs: torch.Tensor,
                             teacher_topk_ids: torch.Tensor) -> float:
    """Greedy EAL diagnostic: mean run of leading positions where the
    drafter's greedy pick matches the teacher's top-1."""
    with torch.no_grad():
        model_top1 = logits.argmax(-1)
        teacher_top1 = teacher_topk_ids[..., 0]
        match = (model_top1 == teacher_top1)
        runs = 0
        for b in range(match.shape[0]):
            run = 0
            for t in range(match.shape[1]):
                if match[b, t]:
                    run += 1
                else:
                    break
            runs += run
        return runs / match.shape[0]


# ---------- stage 4: foresight adapter ----------

class ForesightAdapter(nn.Module):
    """Low-rank head over the drafter's final hidden state predicting
    (a) the target state vector at the next block boundary and (b) the
    anchor token. Drafter frozen; only this trains (EAGLE-4 pattern)."""

    def __init__(self, d_model: int, teacher_dim: int, vocab_size: int,
                 rank: int = 64):
        super().__init__()
        self.state_head = nn.Sequential(
            nn.Linear(d_model, rank, bias=False),
            nn.Linear(rank, teacher_dim, bias=False))
        self.token_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.zeros_(self.state_head[-1].weight)
        nn.init.zeros_(self.token_head.weight)

    def forward(self, hidden: torch.Tensor):
        return self.state_head(hidden), self.token_head(hidden)


def foresight_loss(pred_state, pred_token_logits, teacher_states, next_tokens):
    """CE on the anchor token + cosine alignment on the predicted state."""
    ce = F.cross_entropy(pred_token_logits.flatten(0, 1), next_tokens.flatten())
    a = F.normalize(pred_state.float(), dim=-1)
    b = F.normalize(teacher_states.float(), dim=-1)
    cos = (1 - (a * b).sum(-1)).mean()
    return ce + 0.5 * cos, {"ce": ce.item(), "state_cos": float(cos)}


# ---------- stage 5: draft-forced refinement ----------

@torch.no_grad()
def greedy_continue(model, tokens: torch.Tensor, n_new: int) -> torch.Tensor:
    """Append n_new greedy tokens from the model (draft-forced prefix)."""
    cur = tokens
    for _ in range(n_new):
        logits, _ = model(cur)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat((cur, nxt), dim=1)
    return cur


# ---------- stage 7: fidelity gates ----------

@torch.no_grad()
def top1_mismatch(model_a: nn.Module, model_b: nn.Module,
                  tokens: torch.Tensor) -> tuple[float, float]:
    """Greedy top-1 disagreement of A vs B, and A's self-mismatch baseline
    (B vs B on shifted views is 0 by construction; callers compare the
    quantized drafter against the unquantized one — equality is the gate)."""
    la, _ = model_a(tokens)
    lb, _ = model_b(tokens)
    ta, tb = la.argmax(-1), lb.argmax(-1)
    return (ta != tb).float().mean().item(), 0.0


def fidelity_gate(quantized: nn.Module, reference: nn.Module,
                  tokens: torch.Tensor, acceptance_delta_pt: float = 1.0,
                  _acceptance=None) -> dict:
    """PTQR battery (Section 8.1): quantized top-1 mismatch must equal the
    unquantized drafter's (zero additional loss) and acceptance within
    +-delta. `_acceptance(model, tokens) -> float` is supplied by the
    serving-side harness; without it the acceptance gate is informational."""
    mismatch, _ = top1_mismatch(quantized, reference, tokens)
    report = {"top1_mismatch": mismatch,
              "top1_pass": mismatch == 0.0}
    if _acceptance is not None:
        a_q = _acceptance(quantized, tokens)
        a_r = _acceptance(reference, tokens)
        report.update({"acceptance_quantized": a_q,
                       "acceptance_reference": a_r,
                       "acceptance_pass": abs(a_q - a_r) <= acceptance_delta_pt})
    report["all_pass"] = report["top1_pass"] and report.get(
        "acceptance_pass", True)
    return report
