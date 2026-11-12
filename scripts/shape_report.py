# SPDX-License-Identifier: Apache-2.0
"""Report the effective draft shape vs the RESEARCH.md decided defaults.

Section 6's architecture map fixes the draft at 24 layers x d=1792, 597 Rosa
streams, 6869 KS3 charts, tied embeddings (~0.98B). The recipe YAML may
override those fields; this script prints both sides and the parameter
deltas so drift is visible instead of implicit.

  python scripts/shape_report.py --config configs/flint.yaml
"""

from __future__ import annotations

import argparse
import json

from ksdraft.cli import _load_target_config
from ksdraft.config import KSDraftConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flint.yaml")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    eff, _teacher, _training = _load_target_config(args.config)
    dec = KSDraftConfig()
    fields = ("layers", "d_model", "head_size", "rosa_streams", "rosa_bits",
              "block_len", "vocab_size")
    diff = {f: (getattr(dec, f), getattr(eff, f)) for f in fields
            if getattr(dec, f) != getattr(eff, f)}
    report = {
        "config": args.config,
        "decided": {**{f: getattr(dec, f) for f in fields},
                    "charts": dec.charts,
                    "core_params": dec.total_params()},
        "effective": {**{f: getattr(eff, f) for f in fields},
                      "charts": eff.charts,
                      "core_params": eff.total_params()},
        "overrides": diff,
        "decided_charts": dec.charts,
        "effective_charts": eff.charts,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"decided (RESEARCH §6/§9): {dec.total_params():,} core params, "
          f"{dec.charts} charts")
    print(f"effective ({args.config}): {eff.total_params():,} core params, "
          f"{eff.charts} charts")
    if diff:
        print("overrides vs decided:")
        for k, (a, b) in diff.items():
            print(f"  {k:14s} {a} -> {b}")
    else:
        print("matches the decided shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
