"""Training-path fused head losses (chunked, low-memory).

The reference path materializes full-vocabulary logits for both the
next-token head and the block head, upcasts them to fp32, and runs a second
full-size softmax copy. At V=150k that is ~1.2 GB of transient traffic per
step and a 230M-element weight gradient that dominates the backward.

These functions compute the same math in position chunks:

    logits_c = x_c @ W^T
    lse_c    = logsumexp(logits_c)
    ce_c     = lse_c - logits_c[target]

so the largest live logits tensor is ``chunk x V`` instead of ``N x V``. The
weight-gradient GEMM is identical in FLOPs; the fp32 logits copy, the
``log_softmax`` intermediate, and the fp32 softmax copy disappear.

``linear_soft_ce`` is the distillation twin: the target is the teacher's
renormalized top-k distribution, which is the usual forward KL on the
teacher support (the off-support teacher mass is zero by construction).

Both keep fp32 softmax math and fp16/bf16/fp32 GEMM operands exactly like
autocast ``F.linear`` + ``F.cross_entropy``.
"""

from __future__ import annotations

import torch


def _chunk_logits(x: torch.Tensor, w: torch.Tensor, chunk: int):
    n = x.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        yield s, e, x[s:e] @ w.t()


def _weights(pos_weight: torch.Tensor | None, n: int,
             device) -> torch.Tensor:
    if pos_weight is None:
        return torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    w = pos_weight.float()
    return w / w.sum().clamp_min(1e-9)


_DX_VOCAB_CHUNK = 8192


def _k_chunked_dx(dlog: torch.Tensor, wf: torch.Tensor) -> torch.Tensor:
    """dX = dlog @ W, split over the vocab dim.

    A single K=150k GEMM selects a pathological rocBLAS kernel on gfx908
    (~1.4 TFLOPS, 162 ms at N=512); 8k-wide K chunks run ~10x faster
    (16 ms), and the same split is neutral on backends without the bug.
    """
    n, d = dlog.shape[0], wf.shape[1]
    dx = torch.zeros(n, d, device=dlog.device, dtype=dlog.dtype)
    for v in range(0, wf.shape[0], _DX_VOCAB_CHUNK):
        dx = dx + dlog[:, v:v + _DX_VOCAB_CHUNK] @ wf[v:v + _DX_VOCAB_CHUNK]
    return dx


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, targets, pos_weight, chunk):
        n, d = x.shape
        wf = w if w.dtype == x.dtype else w.to(x.dtype)
        wgt = _weights(pos_weight, n, x.device)
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        per_pos = torch.empty(n, device=x.device, dtype=torch.float32)
        for s, e, logits in _chunk_logits(x, wf, chunk):
            lg = logits.float()
            lse = torch.logsumexp(lg, dim=-1)
            tgt = targets[s:e]
            ce = lse - lg.gather(-1, tgt[:, None]).squeeze(-1)
            per_pos[s:e] = ce
            loss = loss + (wgt[s:e] * ce).sum()
        ctx.save_for_backward(x, wf, w, targets, wgt)
        ctx.chunk = chunk
        ctx.need_dw = ctx.needs_input_grad[1]
        ctx.mark_non_differentiable(per_pos)
        return loss, per_pos

    @staticmethod
    def backward(ctx, grad_loss, grad_per_pos):
        x, wf, w, targets, wgt = ctx.saved_tensors
        n, d = x.shape
        need_dw = getattr(ctx, "need_dw", True)
        dw = (torch.zeros(w.shape, device=w.device, dtype=torch.float32)
              if need_dw else None)
        g = grad_loss.float()
        dlog_all = torch.empty(n, wf.shape[0], device=x.device, dtype=x.dtype)
        for s, e, logits in _chunk_logits(x, wf, ctx.chunk):
            lg = logits.float()
            q = torch.softmax(lg, dim=-1)
            q.scatter_add_(-1, targets[s:e, None], torch.full_like(
                targets[s:e, None], -1.0, dtype=q.dtype))
            q.mul_(wgt[s:e, None] * g)
            dlog_all[s:e] = q.to(x.dtype)
            if need_dw:
                dw = dw + (dlog_all[s:e].t() @ x[s:e]).float()
        dx = _k_chunked_dx(dlog_all, wf)
        return dx, (dw.to(w.dtype) if need_dw else None), None, None, None


class _LinearSoftCE(torch.autograd.Function):
    """Forward KL on the teacher's top-k support, chunked; returns the model
    log-probs on that support (for WTV / diagnostics)."""

    @staticmethod
    def forward(ctx, x, w, ids, probs, pos_weight, chunk):
        n, d = x.shape
        wf = w if w.dtype == x.dtype else w.to(x.dtype)
        k = ids.shape[1]
        wgt = _weights(pos_weight, n, x.device)
        p_t = probs.float()
        p_t = p_t / p_t.sum(-1, keepdim=True).clamp_min(1e-9)
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        topk_logp = torch.empty(n, k, device=x.device, dtype=torch.float32)
        for s, e, logits in _chunk_logits(x, wf, chunk):
            lg = logits.float()
            lse = torch.logsumexp(lg, dim=-1, keepdim=True)
            on = lg.gather(-1, ids[s:e]) - lse
            topk_logp[s:e] = on
            loss = loss + (wgt[s:e] * -(p_t[s:e] * on).sum(-1)).sum()
        ctx.save_for_backward(x, wf, w, ids, p_t, wgt)
        ctx.chunk = chunk
        ctx.need_dw = ctx.needs_input_grad[1]
        return loss, topk_logp

    @staticmethod
    def backward(ctx, grad_loss, grad_topk):
        x, wf, w, ids, p_t, wgt = ctx.saved_tensors
        n, d = x.shape
        need_dw = getattr(ctx, "need_dw", True)
        dw = (torch.zeros(w.shape, device=w.device, dtype=torch.float32)
              if need_dw else None)
        g = grad_loss.float()
        dlog_all = torch.empty(n, wf.shape[0], device=x.device, dtype=x.dtype)
        for s, e, logits in _chunk_logits(x, wf, ctx.chunk):
            lg = logits.float()
            q = torch.softmax(lg, dim=-1)
            # d/dlogit of -sum_k p_k log q_k is q - p (p sparse on support)
            row = torch.zeros_like(q)
            row.scatter_add_(-1, ids[s:e], p_t[s:e])
            dlog = (q - row) * (wgt[s:e, None] * g)
            if grad_topk is not None:
                # logp_k = logit_k - lse  ->  dlogit_j = g_j - (sum g) * q_j
                gt = grad_topk[s:e].float()
                dlog = dlog - gt.sum(-1, keepdim=True) * q
                dlog.scatter_add_(-1, ids[s:e], gt)
            dlog_all[s:e] = dlog.to(x.dtype)
            dw = dw + (dlog_all[s:e].t() @ x[s:e]).float()
        dx = _k_chunked_dx(dlog_all, wf)
        return dx, (dw.to(w.dtype) if need_dw else None), None, None, None, None


def linear_ce(x: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
              pos_weight: torch.Tensor | None = None,
              chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked next-token CE. Returns (loss, per-position CE)."""
    return _LinearCE.apply(x, weight, targets, pos_weight, chunk)


def linear_soft_ce(x: torch.Tensor, weight: torch.Tensor,
                   target_ids: torch.Tensor, target_probs: torch.Tensor,
                   pos_weight: torch.Tensor | None = None,
                   chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked KL to a sparse top-k teacher target.

    Returns (loss, model log-probs on the teacher support)."""
    return _LinearSoftCE.apply(x, weight, target_ids, target_probs,
                               pos_weight, chunk)


def wtv_from_logp(topk_logp: torch.Tensor,
                  teacher_probs: torch.Tensor) -> torch.Tensor:
    """Window total variation on the teacher support, from the model's
    log-probs at those ids (no full-vocab softmax needed)."""
    p_t = teacher_probs.float()
    p_t = p_t / p_t.sum(-1, keepdim=True).clamp_min(1e-9)
    p_m = topk_logp.exp()
    on_diff = (p_m - p_t).abs().sum(-1)
    off_mass = (1.0 - p_m.sum(-1)).clamp(min=0.0)
    return 0.5 * (on_diff + off_mass).mean()


def block_position_weights(n: int, decay: float, device) -> torch.Tensor:
    """Normalized per-offset weights (EAGLE-4's 0.8^i shape, generalized).

    ``decay=1.0`` is the uniform objective; ``decay<1`` shifts supervision
    toward the early offsets that dominate acceptance length."""
    if decay >= 1.0:
        return torch.ones(n, device=device)
    i = torch.arange(n, device=device, dtype=torch.float32)
    return decay ** i
