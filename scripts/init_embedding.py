# SPDX-License-Identifier: Apache-2.0
"""Target-embedding projection for the draft (EAGLE/Quark recipe).

The draft's tied embedding starts from a rank-d projection of the target's
embedding table instead of random noise. The projection keeps the Gram
geometry of the top-d spectrum (P = U_d S_d from the randomized SVD of E), so
token similarities the target learned survive the width change. The trainer
can then freeze the embedding/head (``--freeze-embeddings``), which is the
published EAGLE-3.x and AMD Quark recipe.

  python scripts/init_embedding.py \
      --target /home/curved/models/Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-512k \
      --d-model 1536 --out runs/embedding_init.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def find_tensor(target: Path, name: str) -> tuple[Path, str]:
    index = target / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if name not in weight_map:
            raise KeyError(f"{name} not in {index}")
        return target / weight_map[name], name
    for shard in sorted(target.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            if name in f.keys():
                return shard, name
    raise FileNotFoundError(f"{name} not found under {target}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--tensor", default="model.embed_tokens.weight")
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--rank-oversample", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    target = Path(args.target)
    shard, name = find_tensor(target, args.tensor)
    print(f"loading {name} from {shard.name}", flush=True)
    with safe_open(str(shard), framework="pt") as f:
        emb = f.get_tensor(name).to(device=args.device, dtype=torch.float32)
    print(f"embedding {tuple(emb.shape)}", flush=True)

    d = args.d_model
    q = min(d + args.rank_oversample, min(emb.shape))
    u, s, v = torch.svd_lowrank(emb, q=q, niter=2)
    proj = u[:, :d] * s[:d]                    # [vocab, d] == E @ V_d
    del u, s, v, emb
    torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proj.to(torch.float16).cpu().contiguous(), out)
    print(f"saved {tuple(proj.shape)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
