"""Staged trainer (RESEARCH.md Section 11): Stage 0 warm-up + Stage 2 distill.

Losses:
- block cross-entropy (teacher-forced next-token; the Rosa sweep is causal,
  so every position is a prediction position — anchor density is a data
  property, not a mask property)
- DSpine transfer-space loss: every layer's predicted feature aligned with
  the next token's embedding (cosine)
- top-k KL to the teacher distribution (from the teacher client; absent in
  Stage 0)

Optimization per Section 11: AdamW, cosine schedule with warmup, peak LR
~3e-4, grad-norm clip 1.0. Checkpoints carry config + step + optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from ..config import KSDraftConfig
from ..model import KSDraftModel
from ..teacher import TeacherClient


@dataclass
class TrainConfig:
    steps: int = 50
    seq_len: int = 64
    batch_size: int = 4
    peak_lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 10
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    transfer_weight: float = 0.5
    kl_weight: float = 0.2
    val_every: int = 50
    seed: int = 11


class WindowSampler:
    """Random contiguous windows from a flat token stream (zero Rosa state
    at window start — matches the k-splanifolds byte-LM protocol)."""

    def __init__(self, ids: list[int], seq_len: int, seed: int = 11):
        if len(ids) <= seq_len + 1:
            raise ValueError("corpus too short for the window size")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, len(self.ids) - self.seq_len - 1, (batch_size,),
                            generator=self.g)
        ar = torch.arange(self.seq_len + 1)
        return self.ids[idx[:, None] + ar[None, :]]


def distillation_losses(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    features: list[torch.Tensor],
    embedding_weight: torch.Tensor,
    teacher_probs: torch.Tensor | None = None,
    teacher_topk_ids: torch.Tensor | None = None,
    transfer_weight: float = 0.5,
    kl_weight: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    # logits/teacher signals cover exactly the inp positions (T of them)
    ce = F.cross_entropy(logits.flatten(0, 1), tgt.flatten())

    # transfer-space: per-layer predicted feature vs next-token embedding
    tgt_emb = F.normalize(embedding_weight[tgt].float(), dim=-1)
    ts = 0.0
    for f in features:
        fn = F.normalize(f.float(), dim=-1)
        ts = ts + (1 - (fn * tgt_emb).sum(-1)).mean()
    ts = ts / max(1, len(features))

    loss = ce + transfer_weight * ts
    parts = {"ce": ce.item(), "transfer": float(ts)}

    if teacher_probs is not None and kl_weight > 0:
        logp = F.log_softmax(logits.float(), dim=-1)
        model_top = logp.gather(-1, teacher_topk_ids)
        p_t = teacher_probs.float()
        p_t = p_t / p_t.sum(-1, keepdim=True).clamp(min=1e-9)
        kl = -(p_t * model_top).sum(-1).mean()  # cross-entropy on teacher support
        loss = loss + kl_weight * kl
        parts["kl"] = float(kl)

    return loss, parts


class Trainer:
    def __init__(self, model: KSDraftModel, teacher: TeacherClient | None,
                 sampler: WindowSampler, cfg: TrainConfig,
                 device: str = "cpu", out_dir: str | None = None):
        self.model = model.to(device)
        self.teacher = teacher
        self.sampler = sampler
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)

    def _lr_scale(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return (step + 1) / max(1, c.warmup)
        progress = (step - c.warmup) / max(1, c.steps - c.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return c.min_lr_frac + (1 - c.min_lr_frac) * cos

    def _apply_lr(self, step: int):
        lr = self.cfg.peak_lr * self._lr_scale(step)
        for g in self.opt.param_groups:
            g["lr"] = lr

    def train(self, stage: str = "distill", log=print) -> dict:
        assert stage in ("warmup", "distill")
        model = self.model
        model.train()
        use_teacher = stage == "distill" and self.teacher is not None
        history = []
        for step in range(self.cfg.steps):
            self._apply_lr(step)
            batch = self.sampler.batch(self.cfg.batch_size).to(self.device)
            tokens = batch  # (B, T+1); inputs batch[:, :-1], targets batch[:, 1:]
            states = None
            tp = ti = None
            if use_teacher:
                sig = self.teacher.signals(tokens[:, :-1]).to(
                    self.device, torch.float16 if self.device != "cpu" else torch.float32)
                flat = sig.states.flatten(2) if sig.states is not None else None
                states = flat
                tp, ti = sig.topk_probs, sig.topk_ids
            logits, features = model(tokens[:, :-1], teacher_states=states)
            loss, parts = distillation_losses(
                logits, tokens, features,
                model.embedding.weight, tp, ti,
                self.cfg.transfer_weight,
                self.cfg.kl_weight if use_teacher else 0.0)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            rec = {"step": step, "loss": loss.item(), "grad_norm": float(gn), **parts}
            history.append(rec)
            if step % max(1, self.cfg.steps // 10) == 0 or step == self.cfg.steps - 1:
                log(f"[{stage} {step:4d}] loss {rec['loss']:.4f} "
                    f"ce {rec['ce']:.4f} ts {rec['transfer']:.4f} gn {rec['grad_norm']:.2f}")
            if self.out_dir and ((step + 1) % self.cfg.val_every == 0
                                 or step == self.cfg.steps - 1):
                self.save(step)
        return {"history": history,
                "final": history[-1], "initial": history[0]}

    def save(self, step: int):
        if not self.out_dir:
            raise RuntimeError("no out_dir configured")
        torch.save({
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "train_cfg": vars(self.cfg),
            "config": self.model.config.summary(),
        }, self.out_dir / f"ckpt_{step:06d}.pt")

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        return ckpt
