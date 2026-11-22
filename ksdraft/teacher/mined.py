"""Positioned KV-stream + top-k provider over a mined signal store.

The store (scripts/mine_inkling_signals.py) holds, aligned to the corpus
token stream: per-position 4-tap K/V streams on the deployed int8_block_g128
grid and the teacher's top-k (T=1.0, untruncated). Random training windows
must fetch THEIR spans, so the provider is positioned: it consumes window
start indices alongside the tokens (WindowSampler(return_starts=True)) and
serves causal KV streams for the trunk's tokens — the last T stream
positions ARE the trunk tokens (the reuse attention's stream contract).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import TeacherClient, TeacherSignals


class MinedSignals(TeacherClient):
    def __init__(self, store_dir: str, window: int, seed: int = 11):
        store = Path(store_dir)
        meta = json.loads((store / "meta.json").read_text())
        self.layers = len(meta["taps"])
        self.vocab_size = meta.get("vocab_size", 201024)
        self.topk = int(meta["topk"])
        self.window = int(window)
        self.n_taps = self.layers
        self.seed = seed
        self.k = self._load_stream(store, "k_stream")
        self.v = self._load_stream(store, "v_stream")
        self.ti = torch.load(store / "topk_ids.pt", weights_only=True,
                             mmap=True)
        self.tp = torch.load(store / "topk_probs.pt", weights_only=True,
                             mmap=True)
        self.bounds = torch.load(store / "boundaries.pt", weights_only=True)
        self._starts = None  # set per-batch by the trainer

    def _load_stream(self, store: Path, stem: str) -> torch.Tensor:
        pt = store / f"{stem}.pt"
        if pt.exists():                     # legacy materialized store
            return torch.load(pt, weights_only=True, mmap=True)
        # streaming-disk store: raw fp16 rows (meta carries the shape)
        meta = json.loads((store / "meta.json").read_text())
        rows, taps = int(meta["tokens"]), len(meta["taps"])
        return torch.from_file(
            str(store / f"{stem}.bin"), shared=False, size=rows * taps * 1024,
            dtype=torch.float16).view(rows, taps, 1024)

    def _doc_start(self, pos: torch.Tensor) -> torch.Tensor:
        device = pos.device
        pos = pos.cpu()
        out = self.bounds[torch.searchsorted(self.bounds, pos, right=True)
                          .clamp_min(1) - 1]
        return out.to(device)

    def signals_for_windows(self, tokens: torch.Tensor,
                            starts: torch.Tensor) -> TeacherSignals:
        """tokens (B, T_in) at stream positions [s, s+T_in); returns top-k for
        those positions and KV streams whose LAST T_in entries are the
        tokens' positions (causal window of width min(window, coverage),
        doc-start clamped by repetition like causal_kv_windows)."""
        B, T_in = tokens.shape
        device = tokens.device
        starts = starts.to(device).long()
        W = self.window
        S = min(W + T_in - 1, int(self.k.shape[0]))
        span_lo = starts - (S - T_in)
        doc_lo = self._doc_start(starts.clamp_min(0)).to(device)
        span_lo = torch.maximum(span_lo, doc_lo)
        idx = torch.arange(S, device=device)
        # last T_in positions are the window's tokens; earlier positions walk
        # back their causal span, clamped at the document start (repetition)
        pos = (starts + T_in - 1)[:, None] - (S - 1 - idx)[None, :]
        pos = torch.maximum(pos, span_lo[:, None])
        k = self.k[pos.cpu()]                              # (B, S, L, D)
        v = self.v[pos.cpu()]
        kv = (k.permute(0, 2, 1, 3).to(device, torch.float16),
              v.permute(0, 2, 1, 3).to(device, torch.float16))
        rows = starts[:, None] + torch.arange(T_in, device=device)[None, :]
        ti = self.ti[rows.cpu()].to(device)
        tp = self.tp[rows.cpu()].to(device, torch.float16)
        return TeacherSignals(states=None, topk_ids=ti, topk_probs=tp, kv=kv)

    def signals(self, tokens: torch.Tensor) -> TeacherSignals:
        raise ValueError(
            "MinedSignals is positioned: pass window starts (sampler "
            "return_starts=True) through the trainer's positioned path")
