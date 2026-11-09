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
        assert stage in ("warmup", "distill", "foresight", "refine",
                         "acceptance", "ptqr")
        if stage == "foresight":
            return self._train_foresight(log)
        if stage == "refine":
            return self._train_refine(log)
        if stage == "acceptance":
            return self._train_acceptance(log)
        if stage == "ptqr":
            return self._train_ptqr(log)
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

    # ---------- stages 4-7 (Section 11) ----------

    def _teacher_batch(self, tokens):
        sig = self.teacher.signals(tokens[:, :-1]).to(
            self.device, torch.float16 if self.device != "cpu" else torch.float32)
        flat = sig.states.flatten(2) if sig.states is not None else None
        return flat, sig.topk_probs, sig.topk_ids

    def _train_foresight(self, log) -> dict:
        """Stage 4: drafter frozen; low-rank adapter predicts the target
        state + anchor token (EAGLE-4 freeze-then-head staging)."""
        from .stages import ForesightAdapter, foresight_loss
        if self.teacher is None:
            raise ValueError("foresight stage requires a teacher client")
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        teacher_dim = self.model.teacher_dim or 64
        adapter = ForesightAdapter(cfg.d_model, teacher_dim,
                                   cfg.vocab_size).to(self.device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=self.cfg.peak_lr)
        history = []
        self.model.eval()
        for step in range(self.cfg.steps):
            batch = self.sampler.batch(self.cfg.batch_size).to(self.device)
            with torch.no_grad():
                _, features = self.model(batch[:, :-1])
                hidden = features[-1]  # last-layer drafter state (frozen)
            flat, _, _ = self._teacher_batch(batch)
            pred_state, pred_logits = adapter(hidden.float())
            loss, parts = foresight_loss(
                pred_state, pred_logits, flat, batch[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec = {"step": step, "loss": loss.item(), **parts}
            history.append(rec)
            if step % max(1, self.cfg.steps // 5) == 0:
                log(f"[foresight {step:4d}] loss {rec['loss']:.4f} "
                    f"ce {rec['ce']:.4f} state_cos {rec['state_cos']:.4f}")
        for p in self.model.parameters():
            p.requires_grad_(True)
        return {"history": history, "adapter": adapter,
                "final": history[-1], "initial": history[0]}

    def _train_refine(self, log, n_new: int = 8) -> dict:
        """Stage 5: draft-forced self-refinement — greedy continuations from
        the drafter itself, scored against the teacher on the generated region."""
        from .stages import greedy_continue
        if self.teacher is None:
            raise ValueError("refine stage requires a teacher client")
        model = self.model.train()
        history = []
        for step in range(self.cfg.steps):
            self._apply_lr(step)
            batch = self.sampler.batch(self.cfg.batch_size).to(self.device)
            T = batch.shape[1] - 1
            with torch.no_grad():
                draft = greedy_continue(model, batch[:, : max(4, T - n_new)], n_new)
            T_d = draft.shape[1] - 1
            flat, tp, ti = self._teacher_batch(draft)
            logits, features = model(draft[:, :-1], teacher_states=flat)
            mask = torch.zeros(T_d, dtype=torch.bool, device=self.device)
            mask[T_d - n_new:] = True
            inp, tgt = draft[:, :-1][:, mask], draft[:, 1:][:, mask]
            ce = torch.nn.functional.cross_entropy(
                logits[:, mask].flatten(0, 1), tgt.flatten())
            logp = torch.nn.functional.log_softmax(logits[:, mask].float(), -1)
            p_t = tp[:, mask].float()
            p_t = p_t / p_t.sum(-1, keepdim=True).clamp(min=1e-9)
            kl = -(p_t * logp.gather(-1, ti[:, mask])).sum(-1).mean()
            loss = ce + self.cfg.kl_weight * kl
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            rec = {"step": step, "loss": loss.item(), "ce": ce.item(),
                   "kl": float(kl)}
            history.append(rec)
            if step % max(1, self.cfg.steps // 5) == 0:
                log(f"[refine {step:4d}] loss {rec['loss']:.4f} ce {rec['ce']:.4f}")
        return {"history": history, "final": history[-1], "initial": history[0]}

    def _train_acceptance(self, log) -> dict:
        """Stage 6: Window Total Variation on the teacher's top-k support
        (the T=1.0 regime's operative objective) + CE anchor."""
        from .stages import wtv_loss, expected_accepted_length
        if self.teacher is None:
            raise ValueError("acceptance stage requires a teacher client")
        model = self.model.train()
        history = []
        for step in range(self.cfg.steps):
            self._apply_lr(step)
            batch = self.sampler.batch(self.cfg.batch_size).to(self.device)
            flat, tp, ti = self._teacher_batch(batch)
            logits, features = model(batch[:, :-1], teacher_states=flat)
            wtv = wtv_loss(logits, tp, ti)
            ce = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), batch[:, 1:].flatten())
            loss = wtv + 0.1 * ce
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            eal = expected_accepted_length(logits.detach(), batch[:, 1:], tp, ti)
            rec = {"step": step, "loss": loss.item(), "wtv": wtv.item(),
                   "ce": ce.item(), "eal": eal}
            history.append(rec)
            if step % max(1, self.cfg.steps // 5) == 0:
                log(f"[accept {step:4d}] wtv {rec['wtv']:.4f} ce {rec['ce']:.4f} "
                    f"eal {eal:.2f}")
        return {"history": history, "final": history[-1], "initial": history[0]}

    def _train_ptqr(self, log, dither0: float = 0.5,
                    anneal_frac: float = 0.7) -> dict:
        """Stage 7: PTQR consolidation — geometrically anneal rounding dither
        to zero on the INT8 tier (bit-identical to serving at dither=0), then
        run the fidelity gates against the unquantized drafter."""
        from ..model import to_int8
        from .stages import fidelity_gate
        ref = self.model.eval()
        quantized = to_int8(self.model).train()
        qopt = torch.optim.AdamW(
            [p for p in quantized.parameters() if p.requires_grad],
            lr=self.cfg.peak_lr * 0.1)
        history = []
        anneal_steps = max(1, int(self.cfg.steps * anneal_frac))
        tokens_gate = self.sampler.batch(4).to(self.device)
        for m in quantized.modules():
            if hasattr(m, "_quantize_weight"):
                m._dither = dither0
        for step in range(self.cfg.steps):
            dither = dither0 * (0.5 ** (anneal_steps * step / max(1, self.cfg.steps)))
            for m in quantized.modules():
                if hasattr(m, "_quantize_weight"):
                    m._dither = dither
            batch = self.sampler.batch(self.cfg.batch_size).to(self.device)
            flat, tp, ti = self._teacher_batch(batch) if self.teacher else (None, None, None)
            logits, _ = quantized(batch[:, :-1], teacher_states=flat)
            ce = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), batch[:, 1:].flatten())
            loss = ce
            if tp is not None:
                logp = torch.nn.functional.log_softmax(logits.float(), -1)
                p_t = tp.float()
                p_t = p_t / p_t.sum(-1, keepdim=True).clamp(min=1e-9)
                kl = -(p_t * logp.gather(-1, ti)).sum(-1).mean()
                loss = loss + self.cfg.kl_weight * kl
            qopt.zero_grad(set_to_none=True)
            loss.backward()
            qopt.step()
            rec = {"step": step, "loss": loss.item(), "dither": dither}
            history.append(rec)
            if step % max(1, self.cfg.steps // 5) == 0:
                log(f"[ptqr {step:4d}] loss {rec['loss']:.4f} dither {dither:.4f}")
        # fidelity gates (Section 8.1 battery)
        def _acc(model, toks):
            with torch.no_grad():
                lg, _ = model(toks[:, :-1])
            return (lg.argmax(-1) == toks[:, 1:]).float().mean().item()
        report = fidelity_gate(quantized.eval(), ref, tokens_gate,
                               acceptance_delta_pt=1.0, _acceptance=_acc)
        log(f"[ptqr] fidelity: top1_mismatch={report['top1_mismatch']} "
            f"pass={report['all_pass']}")
        return {"history": history, "quantized": quantized,
                "fidelity": report, "final": history[-1], "initial": history[0]}
