"""Corpus builders (RESEARCH.md Section 11).

Tiers, in the order the trainer consumes them:

1. ``c4``     — general prose windows (per-sample token cap).
2. ``agent``  — tool-call traces (the product workload shape).
3. ``long``   — concatenated documents, tens of thousands of tokens.
4. ``xlong``  — tier-5 needle-in-a-haystack documents: 512k-token C4
   haystacks with a planted access-code fact, half within the last 16k
   tokens (feasible-window retrieval) and half at uniform depth.

``build_corpus`` takes plain text documents plus a ``tokenize`` callable so
the target adapter owns the tokenizer (the dev path uses byte values); the
per-target tokenizer wiring lives in the adapter, not here.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

# ~3 characters per token is the corpus-building convention (RESEARCH.md 11);
# the NIAH haystack is specified in tokens, so build it in characters.
CHARS_PER_TOKEN = 3
NIAH_TOKENS = 512 * 1024
NIAH_CHARS = NIAH_TOKENS * CHARS_PER_TOKEN
RECENT_WINDOW_TOKENS = 16 * 1024
RECENT_WINDOW_CHARS = RECENT_WINDOW_TOKENS * CHARS_PER_TOKEN


@dataclass
class CorpusConfig:
    n_c4: int = 3500
    n_agent: int = 800
    n_long: int = 48
    n_xlong: int = 128
    max_sample: int = 2048      # token cap for tiers 1-2
    long_docs: int = 8          # documents concatenated per tier-3 sample
    val_frac: float = 0.01
    niah_chars: int = NIAH_CHARS


def synthetic_ids(n: int, vocab: int, seed: int = 11) -> list[int]:
    """Random token stream for teacher-free smoke runs."""
    g = random.Random(seed)
    return [g.randrange(vocab) for _ in range(n)]


def _window(text: str, tokenize: Callable[[str], Sequence[int]],
            max_tokens: int, rng: random.Random) -> list[int]:
    """A random contiguous token window from ``text`` (at most max_tokens)."""
    ids = list(tokenize(text))
    if not ids:
        return []
    if max_tokens > 0 and len(ids) > max_tokens:
        start = rng.randrange(0, len(ids) - max_tokens)
        ids = ids[start:start + max_tokens]
    return ids


def _needle_document(rng: random.Random, docs: Sequence[str],
                     index: int, hay_chars: int = NIAH_CHARS):
    """Tier-5 NIAH document: (haystack, project, access code).

    ``index % 2 == 0`` plants the needle inside the final 16k-token window
    (what the serving window can actually retrieve); ``index % 2 == 1``
    plants it at uniform depth. The needle sentence is exactly
    ``The access code for <PROJECT> is <CODE>.``; the caller mines the
    teacher's continuation for the answer.
    """
    if not docs:
        raise ValueError("needle documents need at least one haystack document")
    parts: list[str] = []
    size = 0
    while size < hay_chars:
        doc = docs[rng.randrange(len(docs))]
        parts.append(doc)
        size += len(doc) + 1
    hay = "\n".join(parts)[:hay_chars]

    alpha = string.ascii_uppercase + string.digits
    proj = "PROJECT-" + "".join(rng.choice(alpha) for _ in range(5))
    code = "ZK" + "".join(rng.choice(alpha) for _ in range(6))
    needle = f"The access code for {proj} is {code}."

    if index % 2 == 0:
        # Recent stratum: inside the last retrieval window, not at the edge.
        lo = max(1, len(hay) - RECENT_WINDOW_CHARS + 64)
        hi = len(hay) - 64
        pos = rng.randint(lo, max(lo, hi))
    else:
        # Uniform stratum: anywhere before the final window.
        pos = rng.randint(0, max(0, len(hay) - RECENT_WINDOW_CHARS + 64))
    hay = hay[:pos] + " " + needle + " " + hay[pos:]
    return hay, proj, code


def build_corpus(docs: Sequence[str], traces: Sequence[str],
                 tokenize: Callable[[str], Sequence[int]],
                 cfg: CorpusConfig, rng: random.Random | None = None) -> list[int]:
    """Build the flat token stream for ``ksdraft train`` / ``ksdraft mine``.

    Tiers are emitted in proportion to their counts; the trainer samples
    windows from the flat stream, so ordering is not semantic.
    """
    rng = rng or random.Random(11)
    ids: list[int] = []

    def emit(seq: Iterable[int]) -> None:
        ids.extend(int(t) for t in seq)

    for _ in range(cfg.n_c4):
        emit(_window(docs[rng.randrange(len(docs))], tokenize,
                     cfg.max_sample, rng))
    for _ in range(cfg.n_agent):
        emit(_window(traces[rng.randrange(len(traces))], tokenize,
                     cfg.max_sample, rng))
    for _ in range(cfg.n_long):
        text = "\n".join(docs[rng.randrange(len(docs))]
                         for _ in range(max(1, cfg.long_docs)))
        emit(_window(text, tokenize, cfg.max_sample * cfg.long_docs, rng))
    for i in range(cfg.n_xlong):
        hay, _, _ = _needle_document(rng, docs, i, cfg.niah_chars)
        emit(_window(hay, tokenize, 0, rng))   # tokenizer owns the cap
    return ids


def train_val_split(ids: Sequence[int], val_frac: float,
                    rng: random.Random | None = None) -> tuple[list[int], list[int]]:
    """Contiguous split (the stream is a language-modeling corpus)."""
    ids = list(ids)
    if len(ids) < 2:
        raise ValueError("corpus too short to split")
    cut = int(len(ids) * (1.0 - max(0.0, min(val_frac, 0.5))))
    cut = max(1, min(cut, len(ids) - 1))
    return ids[:cut], ids[cut:]
