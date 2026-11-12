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

import json
import math
import os
import time
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
    # Temp-1.0 objective: window total variation on the teacher's top-k
    # support (the served sampler's operative distribution match).
    wtv_weight: float = 0.5
    # Parallel block drafting: supervise N future tokens from the anchor in
    # one forward (the serving proposal path; 0 disables the block head).
    block_n: int = 0
    block_weight: float = 1.0
    # Head-only phase: train only the block offsets for this many steps so
    # the new head cannot disturb the (already trained) next-token backbone;
    # the backbone unfreezes afterwards.
    block_only_steps: int = 0
    # Block objective position weighting (EAGLE-4 0.8^i shape): offsets that
    # survive verification early dominate acceptance length.
    block_position_decay: float = 0.8
    # DFlash-standard block objective: forward KL(target -> draft) on the
    # block offsets, position-weighted. This is the primary loss for the
    # served proposal surface. In the distill stage the hard block CE drops
    # to the block_ce_weight auxiliary (standard: CE is for hard-label/D-PACE
    # only); block_weight stays the phase-A hard-CE primary.
    block_kl_weight: float = 1.0
    block_ce_weight: float = 0.1
    # DFlash2 pairwise path selector: K-way CE over each offset's unary
    # top-K, teacher-forcing the predecessor. Zero-init pair term, so the
    # objective starts equal to the unary argmax.
    selector_weight: float = 1.0
    # DSpine last-write refinement CE at the block head (paper 4.3/4.5).
    lwr_weight: float = 1.0
    # DSpine Eq. 7 mixes CE with the probability L1 (2 x TV on the support).
    # block_ce_weight is alpha; this weight is (1-alpha) * 2 on the same
    # decayed block positions.
    block_wtv_weight: float = 0.0
    # DSpine layer-wise embedding loss: cosine to the unified transfer space,
    # position decay w_i = exp(-(i)/gamma) (paper Eq. 6).
    transfer_gamma: float = 2.0
    # Teacher-forced predecessor substitution probability (paper 4.5 phases
    # it out); 0 disables.
    spine_sub_prob: float = 0.1
    transfer_space: str | None = None
    # Schedule: "cosine" (reference) or "wsd" (warmup-stable-decay, the
    # speedrun-proven shape for longer runs / late annealing).
    lr_schedule: str = "cosine"
    lr_decay_frac: float = 0.7
    # Tail weight averaging (speedrun #92): an EMA model is written next to
    # the final checkpoint; 0 disables.
    ema_decay: float = 0.0
    # Split optimizer: Muon for the 2D hidden matrices, AdamW for embeddings,
    # norms, biases and the structured tables (torch 2.11 built-in).
    muon: bool = False
    muon_lr: float = 2e-3
    muon_weight_decay: float = 0.01
    # Chunk size for the low-memory head losses.
    head_chunk: int = 128
    # Rolling resume checkpoint + held-out metrics. The checkpoint is
    # written atomically to ckpt_latest.pt every roll_every_sec; the eval
    # cache is a mined teacher cache over held-out windows (KLD + EAL proxy).
    roll_every_sec: float = 300.0
    eval_cache: str | None = None
    eval_windows: int = 512
    eval_batch: int = 32
    # Micro-batches per optimizer step: raises the effective batch without
    # the activation memory of a single large batch (tok/s is unchanged).
    grad_accum: int = 1
    # EAGLE/Quark recipe: the tied embedding (and therefore the head) starts
    # from the target's SVD projection and stays frozen. It removes 30% of
    # the trainable parameters and stops a 7M-token corpus from trying to
    # learn token semantics from scratch.
    freeze_embeddings: bool = False
    # Phase A compute/checkpoint dtype: fp16 (decided), bf16, or fp32. The
    # late PTQR stage converts to the int8 tier for annealing.
    train_dtype: str = "fp16"
    val_every: int = 50
    seed: int = 11


class WindowSampler:
    """Random contiguous windows from a flat token stream (zero Rosa state
    at window start — matches the k-splanifolds byte-LM protocol).

    With ``boundaries`` (document start offsets, as produced by
    :func:`detect_boundaries`), windows are sampled inside a single document
    and never cross a role/document boundary. The speedrun recipe aligns
    batches to EoS for the same reason: a window whose target is the first
    token of an unrelated document teaches the drafter noise.
    """

    def __init__(self, ids: list[int] | torch.Tensor, seq_len: int,
                 seed: int = 11,
                 boundaries: torch.Tensor | list[int] | None = None):
        if len(ids) <= seq_len + 1:
            raise ValueError("corpus too short for the window size")
        self.ids = (ids.detach().to(torch.long).cpu() if isinstance(ids, torch.Tensor)
                    else torch.tensor(ids, dtype=torch.long))
        self.seq_len = seq_len
        self.g = torch.Generator().manual_seed(seed)
        self.starts = None
        if boundaries is not None:
            b = (boundaries.detach().to(torch.long).cpu()
                 if isinstance(boundaries, torch.Tensor)
                 else torch.tensor(boundaries, dtype=torch.long))
            b = b[(b >= 0) & (b < len(self.ids))]
            b = torch.unique(torch.cat([torch.zeros(1, dtype=torch.long), b]))
            self.starts = b

    def _span(self, extra: int) -> int:
        return self.seq_len + 1 + extra

    def batch(self, batch_size: int, extra: int = 0) -> torch.Tensor:
        span = self._span(extra)
        if self.starts is None:
            idx = torch.randint(0, len(self.ids) - span, (batch_size,),
                                generator=self.g)
        else:
            starts = self.starts
            ends = torch.cat([starts[1:], torch.tensor([len(self.ids)])])
            lens = ends - starts
            ok = lens >= span
            starts, lens = starts[ok], lens[ok]
            if starts.numel() == 0:
                raise ValueError("no document is long enough for this window")
            pick = torch.randint(0, starts.numel(), (batch_size,),
                                 generator=self.g)
            max_off = lens[pick] - span
            off = (torch.rand(batch_size, generator=self.g) * (max_off + 1)).long()
            idx = starts[pick] + off
        ar = torch.arange(span)
        return self.ids[idx[:, None] + ar[None, :]]


def detect_boundaries(ids: torch.Tensor,
                      separator: list[int] | torch.Tensor) -> torch.Tensor:
    """Document starts for a corpus formatted with a repeated separator
    (e.g. the chat-template turn marker or ``### Instruction:``)."""
    ids = ids.to(torch.long).cpu()
    sep = torch.as_tensor(separator, dtype=torch.long)
    if sep.numel() == 0 or sep.numel() > ids.numel():
        return torch.zeros(0, dtype=torch.long)
    hit = (ids.unfold(0, sep.numel(), 1) == sep).all(-1).nonzero().flatten()
    return hit


def distillation_losses(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    features: list[torch.Tensor],
    embedding_weight: torch.Tensor,
    teacher_probs: torch.Tensor | None = None,
    teacher_topk_ids: torch.Tensor | None = None,
    transfer_weight: float = 0.5,
    kl_weight: float = 0.2,
    wtv_weight: float = 0.0,
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
        if wtv_weight > 0:
            from .stages import wtv_loss

            wtv = wtv_loss(logits, teacher_probs, teacher_topk_ids)
            loss = loss + wtv_weight * wtv
            parts["wtv"] = float(wtv)

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
        if cfg.freeze_embeddings:
            self.model.embedding.weight.requires_grad_(False)
        self.opt_muon = None
        if cfg.muon and device != "cpu":
            muon_params, adam_params, muon_names = _split_muon(self.model)
            try:
                self.opt_muon = torch.optim.Muon(
                    muon_params, lr=cfg.muon_lr,
                    weight_decay=cfg.muon_weight_decay)
            except (RuntimeError, TypeError) as exc:
                print(f"[opt] Muon unavailable ({exc}); AdamW only", flush=True)
                self.opt_muon = None
                adam_params = [p for p in self.model.parameters()
                               if p.requires_grad]
            if self.opt_muon is not None:
                print(f"[opt] Muon {len(muon_params)} tensors, AdamW "
                      f"{len(adam_params)} (e.g. {muon_names[:2]})", flush=True)
        else:
            adam_params = [p for p in self.model.parameters()
                           if p.requires_grad]
        try:
            # Fused AdamW: one kernel for the whole update (2.4x faster than
            # foreach on this stack); falls back on backends without it.
            self.opt = torch.optim.AdamW(
                adam_params, lr=cfg.peak_lr,
                weight_decay=cfg.weight_decay, fused=True)
        except (RuntimeError, TypeError):
            self.opt = torch.optim.AdamW(
                adam_params, lr=cfg.peak_lr,
                weight_decay=cfg.weight_decay)
        self.ema = None
        if cfg.ema_decay > 0:
            self.ema = {name: p.detach().clone()
                        for name, p in self.model.named_parameters()}
        self._eval_tensors = None
        if cfg.eval_cache:
            c = torch.load(cfg.eval_cache, map_location="cpu",
                           weights_only=True)
            n = int(c.get("filled", c["windows"].shape[0]))
            self._eval_tensors = (c["windows"][:n].long(),
                                  c["topk_ids"][:n].long(),
                                  c["topk_probs"][:n].float())
        self._t0 = time.time()
        self._last_roll = self._t0
        self._step_sub_prob = 0.0

    def _lr_scale(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup:
            return (step + 1) / max(1, c.warmup)
        if str(c.lr_schedule).lower() == "wsd":
            span = max(1, c.steps - c.warmup)
            decay_start = c.warmup + int(span * (1.0 - c.lr_decay_frac))
            if step < decay_start:
                return 1.0
            p = (step - decay_start) / max(1, c.steps - decay_start)
            return c.min_lr_frac + (1 - c.min_lr_frac) * (1.0 - p)
        progress = (step - c.warmup) / max(1, c.steps - c.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return c.min_lr_frac + (1 - c.min_lr_frac) * cos

    def _amp(self):
        dt = str(self.cfg.train_dtype).lower()
        if dt == "fp32" or self.device == "cpu":
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = torch.float16 if dt == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _apply_lr(self, step: int):
        scale = self._lr_scale(step)
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.peak_lr * scale
        if self.opt_muon is not None:
            for g in self.opt_muon.param_groups:
                g["lr"] = self.cfg.muon_lr * scale

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
        block_only = int(self.cfg.block_only_steps) if self.cfg.block_n else 0
        if block_only:
            for name, prm in model.named_parameters():
                prm.requires_grad_(name.startswith("block_delta"))
            print(f"[{stage}] block-head-only warmup for {block_only} steps",
                  flush=True)
        use_teacher = stage == "distill" and self.teacher is not None
        history = []
        accum = max(1, int(self.cfg.grad_accum))
        for step in range(self.cfg.steps):
            self._apply_lr(step)
            # DSpine 4.5: teacher-forced predecessor substitution starts at
            # spine_sub_prob and phases out over the first half of training.
            self._step_sub_prob = self.cfg.spine_sub_prob * max(
                0.0, 1.0 - step / max(1.0, 0.5 * self.cfg.steps))
            if block_only and step == block_only:
                for prm in model.parameters():
                    prm.requires_grad_(True)
                print(f"[{stage}] backbone unfrozen at step {step}", flush=True)
            bn = int(self.cfg.block_n)
            self.opt.zero_grad(set_to_none=True)
            if self.opt_muon is not None:
                self.opt_muon.zero_grad(set_to_none=True)
            for micro in range(accum):
                loss, parts = self._micro_loss(bn, use_teacher)
                (loss / accum).backward()
                if micro == accum - 1:
                    break
                del loss
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            if self.opt_muon is not None:
                self.opt_muon.step()
            if self.ema is not None:
                with torch.no_grad():
                    dcy = self.cfg.ema_decay
                    for name, prm in model.named_parameters():
                        e = self.ema[name]
                        e.mul_(dcy).add_(prm.detach(), alpha=1.0 - dcy)
            rec = {"step": step, "loss": loss.item(), "grad_norm": float(gn), **parts}
            history.append(rec)
            if (self.out_dir and self.cfg.roll_every_sec > 0
                    and time.time() - self._last_roll >= self.cfg.roll_every_sec):
                self.save_rolling(step)
                metrics = self.held_out_eval()
                self._log_metrics(step, rec, metrics)
                log(f"[{stage} {step:4d}] roll@ {time.time() - self._t0:.0f}s "
                    f"ce {rec['ce']:.4f} kld {metrics.get('kld', float('nan')):.4f} "
                    f"eal {metrics.get('eal', float('nan')):.3f}", flush=True)
                self._last_roll = time.time()
            if step % max(1, self.cfg.steps // 10) == 0 or step == self.cfg.steps - 1:
                extra = (f" bce {parts['block_ce']:.4f}"
                         if "block_ce" in parts else "")
                log(f"[{stage} {step:4d}] loss {rec['loss']:.4f} "
                    f"ce {rec['ce']:.4f} ts {rec['transfer']:.4f}{extra} "
                    f"gn {rec['grad_norm']:.2f}")
            if self.out_dir and ((step + 1) % self.cfg.val_every == 0
                                 or step == self.cfg.steps - 1):
                self.save(step)
        return {"history": history,
                "final": history[-1], "initial": history[0]}

    def _micro_loss(self, bn: int, use_teacher: bool):
        """One micro-batch forward/loss (shared by every accumulation slot)."""
        model = self.model
        batch = self.sampler.batch(
            self.cfg.batch_size, extra=max(0, bn - 1)).to(self.device)
        tokens = batch
        states = None
        tp = ti = None
        if use_teacher:
            sig = self.teacher.signals(tokens[:, :-1]).to(
                self.device,
                torch.float16 if self.device != "cpu" else torch.float32)
            states = sig.states.flatten(2) if sig.states is not None else None
            tp, ti = sig.topk_probs, sig.topk_ids
        if bn > 0:
            # Low-memory objective: chunked head losses, no V-sized logits
            # tensors (see kernels/fused_train.py). Teacher KL / WTV run on
            # the same chunks through the sparse-target path.
            from ..kernels.fused_train import (block_position_weights,
                                               linear_ce, linear_soft_ce,
                                               wtv_from_logp)
            ctx = tokens[:, :self.cfg.seq_len]
            main_tokens = tokens[:, :self.cfg.seq_len + 1]
            with self._amp():
                h, features, hs = model.forward_block_hidden(
                    ctx, bn, teacher_states=states,
                    sub_prob=self._step_sub_prob)
            d = h.shape[-1]
            w = model.head.weight
            tgt = tokens[:, self.cfg.seq_len:self.cfg.seq_len + bn]
            ce, _ = linear_ce(h.reshape(-1, d), w,
                              main_tokens[:, 1:].reshape(-1),
                              None, self.cfg.head_chunk)
            ts = transfer_alignment(features, model.transfer_C,
                                    main_tokens[:, 1:], self.cfg.transfer_gamma)
            bw = block_position_weights(
                bn, self.cfg.block_position_decay, h.device).repeat(h.shape[0])
            bce, _ = linear_ce(hs.reshape(-1, d), w, tgt.reshape(-1), bw,
                               self.cfg.head_chunk)
            # Phase A (no teacher): hard block CE is the primary objective.
            # With a teacher, the block forward KL below is primary and the
            # hard CE is only the standard's hard-label auxiliary.
            bw_ce = (self.cfg.block_ce_weight if tp is not None
                     else self.cfg.block_weight)
            loss = ce + self.cfg.transfer_weight * ts + bw_ce * bce
            parts = {"ce": float(ce.detach()),
                     "transfer": float(ts.detach()),
                     "block_ce": float(bce.detach())}
            if self.cfg.selector_weight > 0:
                sel = model.selector_loss(hs, tokens, self.cfg.seq_len - 1, bn)
                loss = loss + self.cfg.selector_weight * sel
                parts["selector"] = float(sel.detach())
            if self.cfg.lwr_weight > 0:
                lwr = model.lwr_loss(hs, tokens, self.cfg.seq_len - 1, bn)
                loss = loss + self.cfg.lwr_weight * lwr
                parts["lwr"] = float(lwr.detach())
            if tp is not None and self.cfg.kl_weight > 0:
                k = tp.shape[-1]
                kl, topk_logp = linear_soft_ce(
                    h.reshape(-1, d), w, ti.reshape(-1, k), tp.reshape(-1, k),
                    None, self.cfg.head_chunk)
                loss = loss + self.cfg.kl_weight * kl
                parts["kl"] = float(kl.detach())
                # Block-offset KL: the speculator proposes from offsets
                # 0..bn-1 of the anchor; supervise each with the teacher
                # distribution at that offset's true position, DFlash-style
                # exponential position weights (normalized inside the loss).
                anchor = self.cfg.seq_len - 1
                if (self.cfg.block_kl_weight > 0 and bn > 0
                        and anchor + bn <= ti.shape[1]):
                    bw_off = block_position_weights(
                        bn, self.cfg.block_position_decay, h.device)
                    ids_off = torch.stack(
                        [ti[:, anchor + j] for j in range(bn)], 1).reshape(-1, k)
                    p_off = torch.stack(
                        [tp[:, anchor + j] for j in range(bn)], 1).reshape(-1, k)
                    w_off = bw_off.repeat(h.shape[0])
                    bkl, b_logp = linear_soft_ce(
                        hs.reshape(-1, d), w, ids_off, p_off, w_off,
                        self.cfg.head_chunk, weight_norm="count")
                    loss = loss + self.cfg.block_kl_weight * bkl
                    parts["block_kl"] = float(bkl.detach())
                    if self.cfg.block_wtv_weight > 0:
                        bwtv = wtv_from_logp(b_logp, p_off)
                        loss = loss + self.cfg.block_wtv_weight * bwtv
                        parts["block_wtv"] = float(bwtv.detach())
                if self.cfg.wtv_weight > 0:
                    wtv = wtv_from_logp(topk_logp, tp.reshape(-1, k))
                    loss = loss + self.cfg.wtv_weight * wtv
                    parts["wtv"] = float(wtv.detach())
            return loss, parts
        with self._amp():
            logits, features = model(tokens[:, :-1], teacher_states=states)
        return distillation_losses(
            logits, tokens, features, model.embedding.weight, tp, ti,
            self.cfg.transfer_weight,
            self.cfg.kl_weight if use_teacher else 0.0,
            self.cfg.wtv_weight if use_teacher else 0.0)

    def save_rolling(self, step: int):
        """Atomic rolling resume checkpoint (model + optimizer + step)."""
        dt = str(self.cfg.train_dtype).lower()
        sd = self.model.state_dict()
        if dt in ("fp16", "bf16"):
            half = torch.float16 if dt == "fp16" else torch.bfloat16
            sd = {k: v.to(half) for k, v in sd.items()}
        tmp = self.out_dir / ".ckpt_latest.tmp"
        torch.save({
            "step": step,
            "model": sd,
            "optimizer": self.opt.state_dict(),
            "optimizer_muon": (self.opt_muon.state_dict()
                               if self.opt_muon is not None else None),
            "train_cfg": vars(self.cfg),
            "config": self.model.config.summary(),
            "rolling": True,
        }, tmp)
        os.replace(tmp, self.out_dir / "ckpt_latest.pt")

    @torch.no_grad()
    def held_out_eval(self) -> dict:
        if self._eval_tensors is None:
            return {}
        amp = (torch.autocast("cuda", dtype=torch.float16)
               if self.device != "cpu" and str(self.cfg.train_dtype) == "fp16"
               else torch.autocast("cuda", enabled=False))
        was_training = self.model.training
        self.model.eval()
        out = held_out_metrics(
            self.model, self._eval_tensors, block_n=int(self.cfg.block_n),
            seq_len=int(self.cfg.seq_len), batch=self.cfg.eval_batch,
            device=self.device, amp=amp,
            limit=self.cfg.eval_windows)
        if was_training:
            self.model.train()
        return out

    def _log_metrics(self, step: int, rec: dict, metrics: dict):
        if not self.out_dir:
            return
        row = {"step": step, "elapsed_s": round(time.time() - self._t0, 1),
               "loss": rec.get("loss"), "ce": rec.get("ce"),
               "block_ce": rec.get("block_ce"), **metrics}
        with open(self.out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def save(self, step: int):
        if not self.out_dir:
            raise RuntimeError("no out_dir configured")
        dt = str(self.cfg.train_dtype).lower()
        sd = self.model.state_dict()
        if dt in ("fp16", "bf16"):
            # Phase-A artifacts are half precision (masters live in the
            # optimizer state during the run).
            half = torch.float16 if dt == "fp16" else torch.bfloat16
            sd = {k: v.to(half) for k, v in sd.items()}
        torch.save({
            "step": step,
            "model": sd,
            "optimizer": self.opt.state_dict(),
            "optimizer_muon": (self.opt_muon.state_dict()
                               if self.opt_muon is not None else None),
            "train_cfg": vars(self.cfg),
            "config": self.model.config.summary(),
        }, self.out_dir / f"ckpt_{step:06d}.pt")
        if self.ema is not None and step == self.cfg.steps - 1:
            ema_sd = {k: v.to(sd[k].dtype) if k in sd else v
                      for k, v in self.ema.items()}
            torch.save({"step": step, "model": ema_sd, "ema": True,
                        "config": self.model.config.summary()},
                       self.out_dir / "ckpt_ema.pt")

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device,
                          weights_only=False)
        # New surfaces (e.g. the parallel block head) start fresh when an
        # older checkpoint predates them; everything else must match.
        missing, _unexpected = self.model.load_state_dict(
            ckpt["model"], strict=False)
        if missing:
            print(f"[load] {len(missing)} new parameter(s) initialized fresh: "
                  f"{missing[:3]}", flush=True)
        if self.opt_muon is not None and ckpt.get("optimizer_muon"):
            try:
                self.opt_muon.load_state_dict(ckpt["optimizer_muon"])
            except ValueError:
                print("[load] Muon state incompatible; restarting it", flush=True)
        try:
            self.opt.load_state_dict(ckpt["optimizer"])
        except ValueError:
            # A new parameter (block head) changes the optimizer's group
            # layout; the backbone weights still resume, Adam state restarts.
            print("[load] optimizer state incompatible with the current "
                  "parameter set; starting optimizer state fresh", flush=True)
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
            # DSpine 4.5: teacher-forced predecessor substitution starts at
            # spine_sub_prob and phases out over the first half of training.
            self._step_sub_prob = self.cfg.spine_sub_prob * max(
                0.0, 1.0 - step / max(1.0, 0.5 * self.cfg.steps))
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
            # DSpine 4.5: teacher-forced predecessor substitution starts at
            # spine_sub_prob and phases out over the first half of training.
            self._step_sub_prob = self.cfg.spine_sub_prob * max(
                0.0, 1.0 - step / max(1.0, 0.5 * self.cfg.steps))
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
            # Geometric anneal across the window, then dither=0 for the
            # remaining 30% so the tier consolidates on the exact serving grid.
            frac = min(1.0, step / anneal_steps)
            dither = dither0 * (0.5 ** (20.0 * frac))
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
        if self.out_dir:
            # The annealed INT8 tier is the deliverable: masters are saved so
            # a deterministic dither=0 re-quantization reproduces the served
            # codes exactly (see export_model(..., dtype="int8")).
            torch.save({
                "step": self.cfg.steps,
                "model": quantized.state_dict(),
                "tier": "int8",
                "dither": 0.0,
                "train_cfg": vars(self.cfg),
                "config": self.model.config.summary(),
                "fidelity": report,
            }, self.out_dir / "ckpt_ptqr.pt")
            log(f"[ptqr] saved {self.out_dir / 'ckpt_ptqr.pt'}")
        return {"history": history, "quantized": quantized,
                "fidelity": report, "final": history[-1], "initial": history[0]}


@torch.no_grad()
def held_out_metrics(model, tensors, block_n: int = 0, seq_len: int = 16,
                     batch: int = 32, device="cuda", amp=None,
                     limit: int | None = None) -> dict:
    """Next-token CE/KLD/EAL plus per-draft-offset distributions.

    ``tensors`` = (windows [N,T+1], teacher ids [N,T,K], teacher probs).
    Direct metrics use the model's next-token head at every position.
    Offset metrics run the parallel block head from an anchor at
    ``seq_len-1`` and compare offset k's distribution with the teacher's
    distribution at position ``seq_len-1+k`` — i.e. the distribution of the
    token the speculator actually proposes at draft step k.
    """
    import torch.nn.functional as F

    if amp is None:
        amp = torch.autocast("cuda", enabled=False)
    w_all, ids_all, p_all = tensors
    n = w_all.shape[0] if limit is None else min(limit, w_all.shape[0])
    w_all, ids_all, p_all = w_all[:n], ids_all[:n], p_all[:n]
    ce_t = kld_t = eal_t = 0.0
    chunks = 0
    off_sum = [0.0] * max(0, block_n)
    off_top = [0.0] * max(0, block_n)
    off_cnt = 0
    for i in range(0, n, max(1, batch)):
        w = w_all[i:i + batch].to(device)
        ids = ids_all[i:i + batch].to(device)
        probs = p_all[i:i + batch].to(device)
        with amp:
            logits, _ = model(w[:, :-1])
        lg = logits.float()
        tgt = w[:, 1:]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
        p_t = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
        logp = F.log_softmax(lg, dim=-1)
        on = logp.gather(-1, ids)
        ent = -(p_t * p_t.clamp_min(1e-12).log()).sum(-1)
        kld = (-(p_t * on).sum(-1) - ent).mean()
        from .stages import expected_accepted_length
        eal = expected_accepted_length(logits, w, probs, ids)
        ce_t += float(ce)
        kld_t += float(kld)
        eal_t += float(eal)
        chunks += 1

        if block_n > 0:
            ctx_len = int(seq_len)
            with amp:
                _h, _f, hs = model.forward_block_hidden(w[:, :ctx_len], block_n)
            weight = model.head.weight
            for k in range(block_n):
                tpos = ctx_len - 1 + k
                if tpos >= ids.shape[1]:
                    continue
                with amp:
                    lg_k = hs[:, k] @ weight.t()
                lg_k = lg_k.float()
                logq = F.log_softmax(lg_k, dim=-1)
                ids_k = ids[:, tpos]
                p_k = probs[:, tpos]
                p_k = p_k / p_k.sum(-1, keepdim=True).clamp_min(1e-9)
                on_k = logq.gather(-1, ids_k)
                ent_k = -(p_k * p_k.clamp_min(1e-12).log()).sum(-1)
                off_sum[k] += float((-(p_k * on_k).sum(-1) - ent_k).mean())
                off_top[k] += float((lg_k.argmax(-1) == ids_k[:, 0]).float().mean())
                off_cnt += 1
    out = {"ce": ce_t / max(1, chunks), "kld": kld_t / max(1, chunks),
           "eal": eal_t / max(1, chunks)}
    if block_n > 0:
        per_chunk = max(1, chunks)
        out["kld_off"] = [x / per_chunk for x in off_sum]
        out["top1_off"] = [x / per_chunk for x in off_top]
    return out


def transfer_alignment(features: list[torch.Tensor], C: torch.Tensor,
                       targets: torch.Tensor,
                       gamma: float = 2.0) -> torch.Tensor:
    """DSpine Eq. 6: layer-wise cosine to the transfer space, decayed.

    ``features`` are the post-injection readouts f_l(h_l,t) [B,T,r]; C is
    the fixed [vocab,r] table. w_i = exp(-i/gamma) over window positions,
    normalized by the weight sum, then averaged over layers.
    """
    tgt = C[targets].float()
    tgt = F.normalize(tgt, dim=-1)
    n = targets.shape[1]
    w = torch.exp(-torch.arange(n, device=targets.device,
                                dtype=torch.float32) / max(gamma, 1e-6))
    w = w / w.sum().clamp_min(1e-9)
    ts = 0.0
    for f in features:
        fn = F.normalize(f.float(), dim=-1)
        per_pos = (1 - (fn * tgt).sum(-1))            # [B,T]
        ts = ts + (per_pos * w[None, :]).sum(-1).mean()
    return ts / max(1, len(features))


def _split_muon(model: nn.Module):
    """Muon owns the dense 2D hidden maps; AdamW keeps embeddings, norms,
    biases and the structured tables (payloads, coefficients, offsets)."""
    muon, adam, names = [], [], []
    skip_tags = ("embedding", "block_delta", "coefficients", "raw_scale",
                 "raw_shift")
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        # Newton-Schulz cost is ~min(m,n)^2 * max(m,n) per matrix; the 5888-wide
        # KS3 maps would cost more than the whole rest of the step, so they
        # stay on AdamW. Rosa/injection/feature maps (<= 4096 wide) qualify.
        is_map = (prm.ndim == 2 and min(prm.shape) >= 2
                  and max(prm.shape) <= 4096
                  and not any(t in name for t in skip_tags))
        if is_map:
            muon.append(prm)
            names.append(name)
        else:
            adam.append(prm)
    return muon, adam, names
