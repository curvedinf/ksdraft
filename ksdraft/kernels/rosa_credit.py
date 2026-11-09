"""Rosa counterfactual-credit training path (the Section 7 pillar).

Drives ksdraft's Rosa semantics — time-shifted d->s projections, one-bit
streams, +payload on a matched one / -payload on a matched zero or no match
(the pinned 1-bit hybrid contract) — through the reference exact-credit
engine (xiaoiecc/qkv-rosa-fast-exact-backward). The engine computes every
single-bit counterfactual credit over the window and maps credits to
projection-space gradients via the Bernoulli map sigmoid(x)(1-sigmoid(x)),
which is what makes the hard-symbol projections trainable.

Mapping onto the engine's API (channels = streams, n_bits = 1):
    emb1 = +payload (learned per stream), emb0 = -payload   -> our contract
    q/k/v = the time-shifted projections (torch, autograd chains)
The engine runs CPU/float64 internally (CUDA tensors incur transfers); the
GPU-native credit kernel is a later milestone — this path is exact and
correct, CPU-bound.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_REPO = Path(os.environ.get(
    "KSDRAFT_ROSA_CREDIT_REPO",
    str(Path.home() / "Projects" / "qkv-rosa-fast-exact-backward")))

_ENGINE_PATH = None


def engine_repo() -> Path | None:
    layer = DEFAULT_REPO / "conv_rosa_transformer" / "rosa_layer.py"
    return DEFAULT_REPO if layer.exists() else None


def engine_available() -> bool:
    return engine_repo() is not None


def _load_fn():
    global _ENGINE_PATH
    repo = engine_repo()
    if repo is None:
        raise RuntimeError(
            "qkv-rosa-fast-exact-backward checkout not found; set "
            "KSDRAFT_ROSA_CREDIT_REPO or clone it next to this project")
    pkg = str(repo / "conv_rosa_transformer")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
        _ENGINE_PATH = pkg
    try:
        from rosa_layer import _HardQKVRosaFn  # noqa: F401  (import builds check)
    except ImportError as e:
        raise RuntimeError(
            f"credit engine import failed ({e}); build it first: "
            f"cd {pkg} && python build.py") from e
    from rosa_layer import _HardQKVRosaFn
    return _HardQKVRosaFn


def rosa_credit_forward(q, k, v, payload, credit_order: str = "second"):
    """q/k/v: (B, T, streams) float tensors (pre-bit projections);
    payload: (streams,) learned scale. Returns (B, T, streams) hard output
    (+payload / -payload per stream) with exact-credit autograd attached."""
    fn = _load_fn()
    payload_ext = payload.view(1, 1, -1)
    emb1 = 1.0 * payload_ext   # +payload on a matched one
    emb0 = -1.0 * payload_ext  # -payload on matched zero / no match
    config = dict(
        credit_order=credit_order, randomized=True, tau=1.0,
        credit_to_logit="bernoulli",
        joint_pairs=1 if credit_order == "second" else 0, joint_rows=32,
        trust_diagonal=0.5 if credit_order == "second" else 0.0,
        joint_row_selection="active",
        backward_mode="exact", approximation_rows=32, approximation_channels=0,
    )
    work: dict = {}
    return fn.apply(q, k, v, emb0, emb1, 1, config, work)
