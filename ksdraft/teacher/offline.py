# SPDX-License-Identifier: Apache-2.0
"""Teacher logits from an in-process vLLM engine (no hidden-state taps).

The acceptance-critical teacher signal is the next-token distribution at
T=1.0 under the served truncation. vLLM's ``prompt_logprobs`` returns the
full log-softmax at every prompt position in one pass, so the client feeds a
token window and keeps, for each position, the top-k support after applying
the serving sampler's top-k/top-p truncation and renormalizing.

Hidden-state taps (the DSpine teacher-conditioning path) are not available
this way; drafts trained with this client use ``teacher_dim=0`` and condition
on their own context.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from . import TeacherClient, TeacherSignals


class OfflineLogitTeacher(TeacherClient):
    def __init__(self, model: str, tensor_parallel_size: int = 4,
                 topk: int = 32, serve_top_p: float = 0.95,
                 serve_top_k: int = 20, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85, seed: int = 0):
        from vllm import LLM

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,          # teacher runs once per mining pass
            seed=seed,
        )
        self.topk = int(topk)
        self.serve_top_p = float(serve_top_p)
        self.serve_top_k = int(serve_top_k)
        self.n_taps = 0
        self._tokenizer = self.llm.get_tokenizer()
        self.vocab_size = len(self._tokenizer)

    @torch.no_grad()
    def signals(self, tokens: torch.Tensor) -> TeacherSignals:
        """tokens: [B, T] (each row is a window ending at the anchor token).

        Returns top-k next-token distributions for the B window positions;
        the caller feeds T+1 tokens and uses signals for positions 0..T-1.
        """
        from vllm import SamplingParams

        rows = tokens.long().cpu().tolist()
        params = SamplingParams(
            max_tokens=1,
            temperature=1.0,
            prompt_logprobs=self.topk,
        )
        outs = self.llm.generate(
            [{"prompt_token_ids": r} for r in rows], params, use_tqdm=False
        )
        B, T = tokens.shape
        k = self.topk
        ids = torch.zeros(B, T, k, dtype=torch.int64)
        probs = torch.zeros(B, T, k, dtype=torch.float16)
        for b, out in enumerate(outs):
            plp = out.prompt_logprobs or []
            # plp[i] scores token i given the prefix; the distribution that
            # predicts token i+1 is plp[i+1].
            for t in range(T):
                entry = plp[t + 1] if t + 1 < len(plp) else None
                if not entry:
                    continue
                cand = sorted(entry.items(),
                              key=lambda kv: kv[1].logprob, reverse=True
                              )[:k]
                p = torch.tensor([math.exp(kv[1].logprob) for kv in cand],
                                 dtype=torch.float32)
                idx = torch.tensor([kv[0] for kv in cand], dtype=torch.int64)
                # serving truncation: top-k then top-p, then renormalize
                if self.serve_top_k and len(p) > self.serve_top_k:
                    p, idx = p[:self.serve_top_k], idx[:self.serve_top_k]
                if self.serve_top_p < 1.0 and len(p) > 1:
                    c = torch.cumsum(p, 0)
                    keep = int((c < self.serve_top_p).sum().item()) + 1
                    keep = max(1, min(keep, len(p)))
                    p, idx = p[:keep], idx[:keep]
                p = p / p.sum().clamp_min(1e-9)
                n = len(p)
                ids[b, t, :n] = idx
                probs[b, t, :n] = p.to(torch.float16)
        return TeacherSignals(states=None, topk_ids=ids, topk_probs=probs)


class CachedTeacher(TeacherClient):
    """Signals mined ahead of time, indexed by window (the distill script
    draws windows and their signals together)."""

    def __init__(self, topk_ids: torch.Tensor, topk_probs: torch.Tensor,
                 start: int = 0):
        self.topk_ids = topk_ids
        self.topk_probs = topk_probs
        self.start = start
        self.n_taps = 0

    def take(self, n: int) -> TeacherSignals:
        i = self.start
        self.start += n
        return TeacherSignals(
            states=None,
            topk_ids=self.topk_ids[i:i + n],
            topk_probs=self.topk_probs[i:i + n],
        )
