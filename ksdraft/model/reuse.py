"""DFly2-style target-KV reuse attention (Tencent Hy4 DFly2 lineage).

The Rosa hard-retrieval mixer is replaced by cross-attention into
target-provided KV: the drafter keeps no KV cache of its own, and each draft
layer consumes one paired tap of the target's cached K/V (per-layer, from
the serving engine or an offline dump). Context arrives as provider-aligned
causal windows — query position t sees only target positions <= t — so the
attention needs no internal mask and the draft round's cost is bounded by
the window, not by context length.

Module contract (matching the repo's additive-init doctrine):

- ``TargetKVAttention``: Q from the draft stream, K/V adapted from the
  target's cached KV; ``out_proj`` is zero-initialized so a fresh block is
  exactly identity.
- ``ReuseBlock``: ``x + attn(LN(x))`` -> KS3 -> DSpine injection — identical
  to :class:`ksdraft.model.block.DraftBlock` with the mixer swapped. Missing
  KV raises; the attention is never silently zeroed.
"""

from __future__ import annotations

import torch
from torch import nn

from .ks3 import KS3
from .spine import AdjacentInjection


class TargetKVAttention(nn.Module):
    """Windowed cross-attention over target-cached K/V.

    Inputs:
        x      (B, T, d)      draft hidden states (queries)
        k_src  (B, T, W, kv_dim)  target K for each query's causal window
        v_src  (B, T, W, kv_dim)  target V for each query's causal window

    ``kv_dim`` is the target's per-layer cached K/V width (e.g. Inkling-2-Mini
    8 KV heads x head_dim 128 = 1024). ``n_kv_heads`` adapted KV heads are
    grouped under ``q_heads`` query heads (GQA; degenerates to MHA when the
    counts match).
    """

    def __init__(self, d_model: int, head_size: int, kv_dim: int,
                 n_kv_heads: int = 8):
        super().__init__()
        if kv_dim <= 0:
            raise ValueError("TargetKVAttention requires kv_dim > 0")
        base = max(1, d_model // head_size)
        self.n_kv = max(1, min(n_kv_heads, base))
        self.n_q = max(self.n_kv, (base // self.n_kv) * self.n_kv)
        self.head_size = head_size
        self.kv_dim = kv_dim
        q_dim = self.n_q * head_size
        kv_out = self.n_kv * head_size
        self.q_proj = nn.Linear(d_model, q_dim, bias=False)
        self.k_adapter = nn.Linear(kv_dim, kv_out, bias=False)
        self.v_adapter = nn.Linear(kv_dim, kv_out, bias=False)
        self.out_proj = nn.Linear(q_dim, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def extra_repr(self) -> str:
        return (f"q_heads={self.n_q}, kv_heads={self.n_kv}, "
                f"head_size={self.head_size}, kv_dim={self.kv_dim}")

    def forward(self, x: torch.Tensor, k_src: torch.Tensor,
                v_src: torch.Tensor, k_scales: torch.Tensor | None = None,
                v_scales: torch.Tensor | None = None) -> torch.Tensor:
        if k_src is None or v_src is None:
            raise ValueError("TargetKVAttention requires target KV "
                             "(k_src, v_src); pass target_kv through the model")
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q, self.head_size)
        if k_src.dtype == torch.int8:
            if k_scales is None or v_scales is None:
                raise ValueError(
                    "int8 KV codes require their group scales "
                    "(k_scales, v_scales) — the cache form travels as a pair")
            return self._forward_codes(x, q, k_src, k_scales, v_src, v_scales)
        # per-layer contract (the trunk slices taps before calling):
        # windows (B, T, W, kv_dim) — provider-aligned, dense; streams
        # (B, S, kv_dim) — full causal streams, banded window inside
        if k_src.dim() == 4:
            return self._forward_windows(x, q, k_src, v_src)
        if k_src.dim() == 3:
            return self._forward_streams(x, q, k_src, v_src)
        raise ValueError(f"target KV must be per-layer windows (B,T,W,D) "
                         f"or streams (B,S,D) or int8 codes+scales; got "
                         f"{tuple(k_src.shape)}")

    def _forward_codes(self, x, q, k_codes, k_scales, v_codes, v_scales):
        """INT8 cache-form streams (B, S, kv_dim): codes + g128 scales.

        The KV adapters consume the codes/scales directly (INT8 GEMM — the
        operand is already the cache content, nothing is dequantized to a
        tensor), and the attention scores run INT8 operands with INT32
        accumulation. Softmax and the value matmul stay fp32 (nonlinear /
        epilogue math), matching the deployed kernel design.
        """
        from ..kernels.gemm import w8a8_gemm
        from ..kernels.quant import quantize_last_dim

        B, S, _ = k_codes.shape
        T = x.shape[1]
        if S < T:
            raise ValueError(f"KV stream ({S}) shorter than the trunk "
                             f"window ({T})")
        W = S - T + 1
        k = self.k_adapter.forward_codes(
            k_codes, k_scales).view(B, S, self.n_kv, self.head_size).float()
        v = self.v_adapter.forward_codes(
            v_codes, v_scales).view(B, S, self.n_kv, self.head_size).float()
        g = self.n_q // self.n_kv
        qg = q.float().view(B, T, self.n_kv, g, self.head_size)
        qflat = qg.permute(0, 2, 1, 3, 4).reshape(
            B, self.n_kv, T * g, self.head_size)
        qc, qs = quantize_last_dim(qflat, group=self.head_size)
        kc, ks = quantize_last_dim(
            k.permute(0, 2, 1, 3).reshape(-1, self.head_size),
            group=self.head_size)
        kperm = k.permute(0, 2, 1, 3)
        kc = kc.view(B, self.n_kv, S, self.head_size)
        ks = ks.view(B, self.n_kv, S, 1)
        scores = torch.empty(B, self.n_kv, T * g, S, device=x.device,
                             dtype=torch.float32)
        from ..kernels.quant import dequantize_last_dim
        for b in range(B):
            for h in range(self.n_kv):
                s_i8 = w8a8_gemm(
                    qc[b, h], qs[b, h], kc[b, h], ks[b, h])
                if torch.is_grad_enabled():
                    # STE: the value is the int8 GEMM; the gradient flows
                    # through the dequantized operands (identity quantizer)
                    q_dq = dequantize_last_dim(
                        qc[b, h].float(), qs[b, h]) + (
                            qflat[b, h] - qflat[b, h].detach())
                    k_dq = dequantize_last_dim(
                        kc[b, h].float(), ks[b, h]) + (
                            kperm[b, h] - kperm[b, h].detach())
                    s_ste = q_dq @ k_dq.t()
                    s_i8 = s_i8.detach() + (s_ste - s_ste.detach())
                scores[b, h] = s_i8
        scores = scores.view(B, self.n_kv, T, g, S).permute(0, 1, 3, 2, 4)
        scores = scores / self.head_size ** 0.5
        mask = self._band_mask(T, S, W, x.device).view(1, 1, 1, T, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        p = scores.softmax(dim=-1)
        out = torch.einsum("bkgts,bskh->bkgth", p, v)
        out = out.permute(0, 3, 1, 2, 4).reshape(
            B, T, self.n_q, self.head_size)
        return self.out_proj(
            out.reshape(B, T, self.n_q * self.head_size).to(x.dtype))

    def _forward_windows(self, x, q, k_src, v_src):
        """Dense provider-aligned per-layer windows (B, T, W, kv_dim) — the
        test and fixture contract. Query t's window must contain only
        positions <= t."""
        B, T, W, _ = k_src.shape
        # unified fp32 attention math: int8-tier projections return fp32 while
        # adapter outputs follow the KV stream dtype (fp16 at serve time)
        q = q.float()
        k = self.k_adapter(k_src).view(B, T, W, self.n_kv, self.head_size).float()
        v = self.v_adapter(v_src).view(B, T, W, self.n_kv, self.head_size).float()
        if self.n_kv != self.n_q:
            rep = self.n_q // self.n_kv
            k = k.repeat_interleave(rep, dim=3)
            v = v.repeat_interleave(rep, dim=3)
        scores = torch.einsum("bthd,btwhd->bthw", q, k) / self.head_size ** 0.5
        p = scores.softmax(dim=-1)
        out = torch.einsum("bthw,btwhd->bthd", p, v)
        return self.out_proj(out.reshape(B, T, self.n_q * self.head_size).to(x.dtype))

    def _forward_streams(self, x, q, k_src, v_src):
        """Causal-window attention over a full per-layer KV stream (B, S, kv_dim).

        The trunk's T tokens are the LAST T stream positions; query t (stream
        index off+t, off = S-T) attends stream positions [off+t-W+1,
        off+t] with W = S-T+1 — a banded mask over the full-stream scores.
        This is the long-context path: memory scales with S, not T*W, so
        W=1024 needs no per-query window materialization. Mathematically
        identical to the windowed path (softmax over the same W entries).
        """
        B, S, _ = k_src.shape
        T = x.shape[1]
        if S < T:
            raise ValueError(f"KV stream ({S}) shorter than the trunk "
                             f"window ({T})")
        W = S - T + 1
        # unified fp32 attention math (see _forward_windows)
        q = q.float()
        k = self.k_adapter(k_src).view(B, S, self.n_kv, self.head_size).float()
        v = self.v_adapter(v_src).view(B, S, self.n_kv, self.head_size).float()
        g = self.n_q // self.n_kv
        qg = q.view(B, T, self.n_kv, g, self.head_size)
        # (B, T, nkv, g, S) — small even at S ~ 1k
        scores = torch.einsum("btkgh,bskh->btkgs", qg, k) \
            / self.head_size ** 0.5
        mask = self._band_mask(T, S, W, x.device).view(1, T, 1, 1, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        p = scores.softmax(dim=-1)
        out = torch.einsum("btkgs,bskh->btkgh", p, v)
        out = out.reshape(B, T, self.n_q, self.head_size)
        return self.out_proj(out.reshape(B, T, self.n_q * self.head_size).to(x.dtype))

    _band_cache: dict = {}

    def _band_mask(self, T: int, S: int, W: int, device) -> torch.Tensor:
        key = (T, S, W, str(device))
        m = self._band_cache.get(key)
        if m is None:
            off = S - T
            q_pos = torch.arange(T, device=device)[:, None] + off
            k_pos = torch.arange(S, device=device)[None, :]
            m = (k_pos <= q_pos) & (k_pos > q_pos - W)
            self._band_cache[key] = m
        return m


class ReuseBlock(nn.Module):
    """The reuse variant of DraftBlock: target-KV attention + KS3 + DSpine.

        h    = x + TargetKVAttention(LN_attn(x), k_tap, v_tap)
        y    = h + KS3(LN_ffn(h))
        y, f = AdjacentInjection(y, prev_tokens)

    Init doctrine matches DraftBlock: every additive update starts at zero
    (attention out proj, KS3 up, injection W), so a fresh block is exactly
    identity.
    """

    def __init__(self, config, layer: int, spine_core, rank: int,
                 transfer_C: torch.Tensor | None = None,
                 R_init: torch.Tensor | None = None):
        super().__init__()
        d = config.d_model
        self.ln_attn = nn.LayerNorm(d)
        self.attn = TargetKVAttention(d, config.head_size, config.kv_dim,
                                      config.n_kv_heads)
        self.injection = AdjacentInjection(d, rank, spine_core, transfer_C,
                                           R_init)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = KS3(d, config.ks3_charts or None)

    def forward(
        self,
        x: torch.Tensor,
        prev_tokens: torch.Tensor | None = None,
        target_k: torch.Tensor | None = None,
        target_v: torch.Tensor | None = None,
        sub_prob: float = 0.0,
        k_scales: torch.Tensor | None = None,
        v_scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_k is None or target_v is None:
            raise ValueError("ReuseBlock requires target KV "
                             "(target_k, target_v) at every forward")
        h = x + self.attn(self.ln_attn(x), target_k, target_v,
                          k_scales=k_scales, v_scales=v_scales)
        y = h + self.ffn(self.ln_ffn(h))
        y, feature = self.injection(y, prev_tokens, sub_prob)
        return y, feature
