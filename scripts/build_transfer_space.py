# SPDX-License-Identifier: Apache-2.0
"""Build DSpine's unified transfer space from the target LM head.

Paper (arXiv 2609.36173, App. A): with the row-normalized LM head Ebar and
its vocabulary mean mu, keep the top-r eigenvectors P_r / eigenvalues L_r of
the centered Gram matrix, and define

    C_v = normalize(L_r^-1/2 P_r^T (Ebar_v - mu))          [vocab, r]

R_l is initialized with P_r^T. Our draft width is 1536 while the target is
2048, so the readout init is mapped through the embedding SVD basis V
(P = E V): R_init = P_r^T V, shape [r, 1536].

  python scripts/build_transfer_space.py \
      --target /home/curved/models/Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-512k \
      --d-model 1536 --rank 256 --out runs/transfer_space.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def find_tensor(target: Path, name: str) -> Path:
    index = target / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        return target / wm[name]
    for shard in sorted(target.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            if name in f.keys():
                return shard
    raise FileNotFoundError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--tensor", default="model.embed_tokens.weight")
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    shard = find_tensor(Path(args.target), args.tensor)
    print(f"loading {args.tensor} from {shard.name}", flush=True)
    with safe_open(str(shard), framework="pt") as f:
        e = f.get_tensor(args.tensor).to(device=args.device, dtype=torch.float32)
    emb = torch.nn.functional.normalize(e, dim=-1)
    mu = emb.mean(0)
    x = emb - mu
    gram = x.t() @ x                                  # [H, H]
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)[:args.rank]
    lam = evals[order].clamp_min(1e-12)
    p = evecs[:, order]                               # [H, r]
    c = torch.nn.functional.normalize(
        (x @ p) * lam.rsqrt()[None, :], dim=-1)       # [vocab, r]

    # embedding right-singular basis V [H, d] for the readout init
    # (eigh of the HxH Gram is far cheaper than an SVD of [150k, H])
    ge = e.t() @ e
    ev, evec = torch.linalg.eigh(ge)
    order_e = torch.argsort(ev, descending=True)[:args.d_model]
    v = evec[:, order_e]                              # [H, d]
    r_init = (p.t() @ v)                              # [r, d]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"C": c.to(torch.float16).cpu(),
                "R_init": r_init.to(torch.float32).cpu(),
                "rank": args.rank, "d_model": args.d_model,
                "target_hidden": int(e.shape[1]),
                "eigenvalues": lam.to(torch.float32).cpu()}, out)
    print(f"saved C {tuple(c.shape)} R_init {tuple(r_init.shape)} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
