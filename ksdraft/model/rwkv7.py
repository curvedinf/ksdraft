"""RWKV-7 generalized delta-rule time-mixing mixer — reference port.

Ported faithfully from k-splanifolds experiments/rwkv7_ks2/run.py (TimeMix +
recurrence), which encodes the documented design: six learned token shifts,
normalized removal keys, key modulation, first-layer value residual carried
across blocks, output gate, bonus readout, group normalization.

Deviations from the source (documented):
- head width is configurable (source hardcodes 32; the decided ksdraft shape
  uses head_size 64), with the zigzag init generalized to the head width.
- value_control exists on EVERY layer (source: all but layer 0) so per-layer
  parameter counts are uniform; first_value is still set by block 0 and
  carried through the stack exactly as in the source Model.

Reference tier: the position loop runs in Python (O(T) steps over a
(B, H, W, W) state). The INT8 tier replaces this with fused/chunked kernels.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def delta_rule_recurrence(
    read: torch.Tensor, decay: torch.Tensor, key: torch.Tensor,
    value: torch.Tensor, normalized_key: torch.Tensor, rate: torch.Tensor,
) -> torch.Tensor:
    """RWKV-7 generalized delta rule.

    All inputs (B, T, H, W). State (B, H, W, W):
        removal      = state @ normalized_key_t
        state        = state * decay_t - removal * (normalized_key_t * rate_t)
        state       += value_t x key_t
        out_t        = state @ read_t
    """
    batch, length, heads, width = read.shape
    state = torch.zeros(batch, heads, width, width, dtype=read.dtype, device=read.device)
    outputs = []
    for pos in range(length):
        removal = torch.matmul(state, normalized_key[:, pos].unsqueeze(-1))
        state = state * decay[:, pos].unsqueeze(-2) \
            - removal * (normalized_key[:, pos] * rate[:, pos]).unsqueeze(-2)
        state = state + value[:, pos].unsqueeze(-1) * key[:, pos].unsqueeze(-2)
        outputs.append(torch.matmul(state, read[:, pos].unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, 1)


class _Controller(nn.Module):
    """Low-rank data-dependent controller (zero down / orthogonal up init)."""

    def __init__(self, width: int, rank: int, activation: str):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.activation = activation
        nn.init.zeros_(self.down.weight)
        nn.init.orthogonal_(self.up.weight, gain=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.down(inputs)
        if self.activation == "tanh":
            hidden = hidden.tanh()
        elif self.activation == "sigmoid":
            hidden = hidden.sigmoid()
        return self.up(hidden)


class RWKV7TimeMix(nn.Module):
    def __init__(self, width: int, layer: int, layers: int, head_size: int = 64):
        super().__init__()
        self.width = width
        self.head_size = head_size
        self.heads = width // head_size

        fraction = torch.arange(width).float() / width
        depth = 1 - layer / layers
        for name, power in zip(("read", "decay", "key", "value", "rate", "gate"),
                               (0.2, 0.9, 0.7, 0.7, 0.9, 0.2)):
            self.register_parameter(
                "mix_" + name, nn.Parameter(1 - fraction.pow(power * depth)))
        h = head_size
        zigzag = (torch.arange(width) % h - (h - 1) / 2) / ((h - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        linear = torch.linspace(-0.5, 0.5, width)
        self.decay_bias = nn.Parameter(
            -6 + 6 * torch.linspace(0, 1, width).pow(1 + (layer / (layers - 1)) ** 0.3)
            + 0.5 + zigzag * 2.5)
        self.rate_bias = nn.Parameter(-0.19 + zigzag * 0.3 + linear * 0.4)
        self.value_bias = nn.Parameter(0.73 - linear * 0.4)
        self.key_scale = nn.Parameter(0.71 - linear * 0.1)
        self.key_rate = nn.Parameter(torch.full((width,), 1.02))
        self.bonus = nn.Parameter(torch.full((self.heads, h), -0.04))
        self.decay_control = _Controller(width, 32, "tanh")
        self.rate_control = _Controller(width, 32, "linear")
        self.gate_control = _Controller(width, 64, "sigmoid")
        self.value_control = _Controller(width, 32, "linear")

        for name, scale in (("read", 0.5), ("key", 0.05), ("value", 0.5)):
            module = nn.Linear(width, width, bias=False)
            nn.init.uniform_(module.weight, -scale / math.sqrt(width), scale / math.sqrt(width))
            setattr(self, name, module)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.norm = nn.GroupNorm(self.heads, width, eps=64e-5)

    def extra_repr(self) -> str:
        return (f"width={self.width}, heads={self.heads}x{self.head_size}, "
                f"layer-dependent init")

    def forward(self, inputs: torch.Tensor, first_value: torch.Tensor | None):
        batch, length, width = inputs.shape
        shift = F.pad(inputs[:, :-1], (0, 0, 1, 0)) - inputs
        names = ("read", "decay", "key", "value", "rate", "gate")
        mixed = {n: inputs + shift * getattr(self, "mix_" + n) for n in names}

        read = self.read(mixed["read"])
        decay = torch.exp(
            -torch.sigmoid(self.decay_bias + self.decay_control(mixed["decay"]))
            * math.exp(-0.5))
        rate = torch.sigmoid(self.rate_bias + self.rate_control(mixed["rate"]))
        key, value = self.key(mixed["key"]), self.value(mixed["value"])
        if first_value is None:
            first_value = value
        else:
            value = value + (first_value - value) * torch.sigmoid(
                self.value_bias + self.value_control(mixed["value"]))

        shape = (batch, length, self.heads, self.head_size)
        normalized_key = F.normalize((key * self.key_scale).reshape(shape), dim=-1)
        key = key * (1 + (rate - 1) * self.key_rate)
        read, decay, key, value, rate = [
            item.reshape(shape) for item in (read, decay, key, value, rate)]
        output = delta_rule_recurrence(read, decay, key, value, normalized_key, rate)
        output = self.norm(output.reshape(batch * length, width)).reshape(batch, length, width)
        output = output + ((read * key * self.bonus).sum(-1, keepdim=True) * value).reshape(
            batch, length, width)
        return self.output(output * self.gate_control(mixed["gate"])), first_value
