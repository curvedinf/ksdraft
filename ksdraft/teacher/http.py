# SPDX-License-Identifier: Apache-2.0
"""Live teacher signals over the served recipe's HTTP API.

The fork's in-process ``vllm.LLM`` cannot construct an engine for the flint
config (ModelConfig / CacheConfig failures), so the live teacher used by the
refine and acceptance stages is the running API server. One batched
``/completions`` request scores every window in a training micro-batch with
``prompt_logprobs``; position i of the response gives the distribution that
predicts token i+1, exactly like the mining cache.
"""

from __future__ import annotations

import json
import math
import urllib.request

import torch

from . import TeacherClient, TeacherSignals


def parse_entry(entry, topk: int, serve_top_k: int, serve_top_p: float):
    """Top-k ids/probs from one prompt_logprobs position, serving-truncated."""
    if not entry:
        return None
    cand = sorted(entry.items(), key=lambda kv: kv[1]["logprob"],
                  reverse=True)[:topk]
    p = torch.tensor([math.exp(kv[1]["logprob"]) for kv in cand],
                     dtype=torch.float32)
    idx = torch.tensor([int(kv[0]) for kv in cand], dtype=torch.int64)
    if serve_top_k and len(p) > serve_top_k:
        p, idx = p[:serve_top_k], idx[:serve_top_k]
    if serve_top_p < 1.0 and len(p) > 1:
        c = torch.cumsum(p, 0)
        keep = int((c < serve_top_p).sum().item()) + 1
        keep = max(1, min(keep, len(p)))
        p, idx = p[:keep], idx[:keep]
    return idx, p / p.sum().clamp_min(1e-9)


class HTTPTeacher(TeacherClient):
    def __init__(self, base_url: str, model: str = "flint",
                 api_key: str = "", topk: int = 32,
                 serve_top_p: float = 0.95, serve_top_k: int = 20,
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.topk = int(topk)
        self.serve_top_p = float(serve_top_p)
        self.serve_top_k = int(serve_top_k)
        self.timeout = float(timeout)
        self.n_taps = 0

    def signals(self, tokens: torch.Tensor) -> TeacherSignals:
        # prompt_logprobs cannot score position 0 (no prefix). Append one
        # dummy token: it is causal, so it only adds the last real position's
        # next-token distribution (which the T-token prompt lacks).
        rows = [r + [0] for r in tokens.long().cpu().tolist()]
        body = {"model": self.model, "prompt": rows, "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": min(self.topk, 20)}
        req = urllib.request.Request(
            self.base_url + "/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        B, T = tokens.shape
        k = self.topk
        ids = torch.zeros(B, T, k, dtype=torch.int64)
        probs = torch.zeros(B, T, k, dtype=torch.float16)
        for b, choice in enumerate(out["choices"]):
            plp = choice.get("prompt_logprobs") or []
            for t in range(T):
                parsed = parse_entry(
                    plp[t + 1] if t + 1 < len(plp) else None,
                    k, self.serve_top_k, self.serve_top_p)
                if parsed is None:
                    continue
                ci, cp = parsed
                ids[b, t, :len(ci)] = ci
                probs[b, t, :len(cp)] = cp
        return TeacherSignals(states=None, topk_ids=ids, topk_probs=probs)
