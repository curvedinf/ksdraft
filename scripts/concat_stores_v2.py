"""Concatenate per-pack store-v2 dirs into one training store.

Rows are row-major (rows, taps, dim) in every file, so codes/scales files
concatenate byte-for-byte; top-k tensors concatenate along rows; boundaries
become the cumulative pack starts (each pack is an independent document).
"""
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: concat_stores_v2.py OUT DIR1 DIR2 ...", file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    srcs = [Path(p) for p in sys.argv[2:]]
    out.mkdir(parents=True, exist_ok=True)
    rows_total = 0
    bounds = [0]
    metas = []
    for src in srcs:
        meta = json.loads((src / "meta.json").read_text())
        metas.append(meta)
        rows_total += int(meta["tokens"])
        bounds.append(rows_total)
    for name in ("k_codes.bin", "k_scales.bin", "v_codes.bin", "v_scales.bin"):
        with open(out / name, "wb") as f:
            for src in srcs:
                with open(src / name, "rb") as g:
                    while True:
                        chunk = g.read(1 << 26)
                        if not chunk:
                            break
                        f.write(chunk)
    ti = torch.cat([torch.load(s / "topk_ids.pt", weights_only=True)
                    for s in srcs], dim=0)
    tp = torch.cat([torch.load(s / "topk_probs.pt", weights_only=True)
                    for s in srcs], dim=0)
    torch.save(ti, out / "topk_ids.pt")
    torch.save(tp, out / "topk_probs.pt")
    # boundaries: every pack start except the duplicated tail entry
    b = torch.tensor(bounds[:-1], dtype=torch.long)
    torch.save(b, out / "boundaries.pt")
    m0 = metas[0]
    (out / "meta.json").write_text(json.dumps({
        "source": m0.get("source"), "corpus": m0.get("corpus"),
        "tokens": rows_total, "taps": m0["taps"], "topk": m0["topk"],
        "format": m0["format"], "kv_grid": m0["kv_grid"],
        "codes_dtype": "int8", "scales_dtype": "fp16", "group": 128,
        "packs": len(srcs), "pack_rows": bounds[1:],
    }, indent=2))
    print(f"concatenated {len(srcs)} packs -> {out} ({rows_total:,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
