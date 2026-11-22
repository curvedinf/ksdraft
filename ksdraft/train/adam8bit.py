"""Block-wise 8-bit AdamW states; update math aligned to modded-nanogpt.

Reference: ~/modded-nanogpt mi300/optim/anvil.py ``_adam_update_step`` (the
project's optimizer reference, adopted 2026-11-22). The step matches its
form:

    update = m / (sqrt(v) + eps) * lr * sqrt(1-b2^t) / (1-b1^t)
             # eps added to raw sqrt(v); both bias corrections live in the
             # step size
    mask = (update * p) > 0                      # CAUTIOUS weight decay:
    update += p * mask * lr^2 * wd * wd_mul      # sign-masked and lr^2-scaled
    p -= update

Per-group config (param_groups entries): ``betas`` (default (0.9, 0.99),
the reference's non-head groups; the head/embedding group carries
(0.5, 0.95)) and ``wd_mul`` (default 0.0 — the reference applies no decay
outside the head, which uses 150).

Memory decision kept on top of the reference math: the moments are stored
as block-wise int8 (m and sqrt(v)) with per-block fp32 absmax scales
(2048-element blocks), dequantized to fp32 temporaries inside ``step()``.
Two guards this storage requires:

- sqrt(v) rather than v is quantized (half the log-spread; a naive v
  quantizer diverges when a coordinate's v rounds to zero inside a
  large-absmax block while m stays resolved);
- the preconditioned ratio is clamped to +-``ratio_clip`` (default 32 ~
  1/sqrt(1-beta2)) so a quantization artifact cannot produce an unbounded
  step.

Consequently eps stays 1e-8, NOT the reference's 1e-10 — a sub-grid eps is
incoherent under 8-bit sqrt(v) storage.

Parameters (the fp32 masters) and gradients stay fp32; autocast accumulates
gradients into the leaf dtype. Legacy fp32 ``m``/``v`` states passed to
``load_state_dict`` are quantized on load, so checkpoints written by the
plain fp32 AdamW resume directly.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

DEFAULT_BLOCK = 2048
DEFAULT_RATIO_CLIP = 32.0


def quantize_state(x: torch.Tensor, block: int = DEFAULT_BLOCK
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 tensor -> (int8 codes, fp32 per-block scales)."""
    flat = x.detach().float().reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1)
    scale = (absmax / 127.0).clamp_min(1e-12)
    codes = torch.round(blocks / scale.unsqueeze(1)).clamp_(-127.0, 127.0)
    return codes.to(torch.int8), scale


def dequantize_state(codes: torch.Tensor, scales: torch.Tensor, n: int,
                     like: torch.Tensor) -> torch.Tensor:
    """(int8 codes, fp32 scales) -> fp32 tensor of the original numel."""
    flat = codes.float() * scales.unsqueeze(1)
    return flat.reshape(-1)[:n].view_as(like)


class AdamW8bit(Optimizer):
    """AdamW with block-wise int8 m and sqrt(v) storage; reference math."""

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.99),
                 eps: float = 1e-8, weight_decay: float = 1e-2,
                 block_size: int = DEFAULT_BLOCK,
                 ratio_clip: float = DEFAULT_RATIO_CLIP):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if ratio_clip < 1.0:
            raise ValueError("ratio_clip must be >= 1")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        wd_mul=0.0)
        super().__init__(params, defaults)
        self.block_size = block_size
        self.ratio_clip = ratio_clip

    def _zeros(self, like: torch.Tensor):
        return quantize_state(torch.zeros_like(like), self.block_size)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            wd, wd_mul = group["weight_decay"], group.get("wd_mul", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m_codes"], state["m_scales"] = self._zeros(p)
                    state["s_codes"], state["s_scales"] = self._zeros(p)
                state["step"] += 1
                t = state["step"]
                n = p.numel()
                m = dequantize_state(state["m_codes"], state["m_scales"],
                                     n, p)
                s = dequantize_state(state["s_codes"], state["s_scales"],
                                     n, p)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v' = beta2 * s^2 + (1-beta2) * g^2, stored again as sqrt(v)
                s = s.mul_(s).mul_(beta2) \
                    .addcmul_(grad, grad, value=1 - beta2).sqrt_()
                # reference form: eps on the raw sqrt(v), corrections in the
                # step size (anvil.py _adam_update_step)
                step_size = lr * math.sqrt(bc2) / bc1
                ratio = (m / s.add_(eps)).clamp_(-self.ratio_clip,
                                                 self.ratio_clip)
                update = ratio * step_size
                if wd != 0 and wd_mul != 0:
                    mask = (update * p) > 0          # cautious decay mask
                    update = update + p * mask * (lr * lr * wd * wd_mul)
                p.sub_(update)
                state["m_codes"], state["m_scales"] = quantize_state(
                    m, self.block_size)
                state["s_codes"], state["s_scales"] = quantize_state(
                    s, self.block_size)
        return loss

    def state_bytes(self) -> int:
        """Bytes held by the quantized states (diagnostics/tests)."""
        total = 0
        for state in self.state.values():
            if "m_codes" not in state:
                continue
            for key in ("m_codes", "s_codes"):
                total += state[key].numel() * state[key].element_size()
            total += (state["m_scales"].numel()
                      + state["s_scales"].numel()) * 4
        return total

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Legacy fp32 states (written by the plain AdamW) quantize on load;
        # v arrives as the second moment, storage wants its square root.
        for state in self.state.values():
            if "m" in state and state["m"].dtype.is_floating_point:
                state["m_codes"], state["m_scales"] = quantize_state(
                    state.pop("m"), self.block_size)
            if "v" in state and state["v"].dtype.is_floating_point:
                s = state.pop("v").clamp_min(0.0).sqrt_()
                state["s_codes"], state["s_scales"] = quantize_state(
                    s, self.block_size)
