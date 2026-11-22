# SPDX-License-Identifier: Apache-2.0
"""Target-embedding projection for the draft (EAGLE/Quark recipe).

Two modes:

``--copy`` (the Inkling recipe): the draft width equals the target hidden
size, so the target's own table is copied verbatim — no projection. The
serving side binds the same table (shared embedding, DFly2 pattern), so the
training copy must stay raw: normalizing rows here would mismatch the table
the deployed draft actually reads. Combine with ``--freeze-embeddings``.

default (the Flint recipe): the draft's tied embedding starts from a rank-d
projection of the target's table instead of random noise. The projection
keeps the Gram geometry of the top-d spectrum (P = U_d S_d from the
randomized SVD of E), so token similarities the target learned survive the
width change. The trainer can then freeze the embedding/head
(``--freeze-embeddings``), which is the published EAGLE-3.x and AMD Quark
recipe.

  # direct copy (dims must match exactly) — the draft's INPUT table
  python scripts/init_embedding.py --copy \
      --target ~/models/Inkling-2-Mini-PTQR-W4A8-G128 \
      --tensor model.llm.embed.weight \
      --out runs/embedding_init.pt

  # dequantized W4 unembed — the draft's untied HEAD (deployed grid)
  python scripts/init_embedding.py --unembed \
      --target ~/models/Inkling-2-Mini-PTQR-W4A8-G128 \
      --out runs/head_init.pt

  # rank-d projection (width differs)
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


def dequant_unembed(weight: torch.Tensor, scale: torch.Tensor,
                    zero: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    """Recover the deployed W4 lm_head grid from the checkpoint layout.

    Matches aiter's ``dequantize_affine_weights`` plain (checkpoint) path:
    U8 [V, K/2] packed nibbles, LOW nibble first; asymmetric affine
    ``w = (code - zero) * scale`` with fp16 per-row GS128 scale and zero.
    """
    n = weight.shape[-2]
    k = scale.shape[-1] * 128
    codes = torch.stack((weight & 15, weight >> 4), dim=-1)
    codes = codes.reshape(n, -1, 128).float()
    values = codes - zero.unsqueeze(-1).float()
    return (values * scale.unsqueeze(-1).float()).reshape(n, k).to(dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--tensor", default="model.embed_tokens.weight")
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--rank-oversample", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-normalize", action="store_true",
                    help="keep the raw projection scale (the default row-"
                         "normalizes so the frozen tied head starts at "
                         "cosine-scale logits)")
    ap.add_argument("--copy", action="store_true",
                    help="copy the target table verbatim (requires the "
                         "draft width == target hidden size; rows stay raw "
                         "so a shared-embedding serving bind matches)")
    ap.add_argument("--unembed", action="store_true",
                    help="dequantize the target's W4 lm_head "
                         "(model.llm.unembed.{weight,weight_scale,"
                         "weight_zero}) into the exact deployed head grid")
    ap.add_argument("--rows-per-chunk", type=int, default=16384)
    args = ap.parse_args()

    target = Path(args.target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.unembed:
        shard, _ = find_tensor(target, "model.llm.unembed.weight_scale")
        with safe_open(str(shard), framework="pt") as f:
            w = f.get_slice("model.llm.unembed.weight")
            sc = f.get_slice("model.llm.unembed.weight_scale")
            zr = f.get_slice("model.llm.unembed.weight_zero")
            n, k2 = w.get_shape()
            k = sc.get_shape()[-1] * 128
            if (k2 * 2, k) != (k, k) or k % 128:
                raise SystemExit(f"unexpected unembed layout {w.get_shape()}")
            result = torch.empty((n, k), dtype=torch.float16)
            for lo in range(0, n, args.rows_per_chunk):
                hi = min(lo + args.rows_per_chunk, n)
                result[lo:hi] = dequant_unembed(
                    w[lo:hi], sc[lo:hi], zr[lo:hi])
            torch.save(result.contiguous(), out)
            print(f"dequantized head {tuple(result.shape)} -> {out}",
                  flush=True)
        return 0

    shard, name = find_tensor(target, args.tensor)
    print(f"loading {name} from {shard.name}", flush=True)
    with safe_open(str(shard), framework="pt") as f:
        emb = f.get_tensor(name).to(device=args.device, dtype=torch.float32)
    print(f"embedding {tuple(emb.shape)}", flush=True)

    if args.copy:
        if emb.shape[1] != args.d_model:
            raise SystemExit(
                f"--copy needs the draft width ({args.d_model}) to equal the "
                f"target hidden size ({emb.shape[1]}); use the SVD projection")
        proj = emb
    else:
        d = args.d_model
        q = min(d + args.rank_oversample, min(emb.shape))
        u, s, v = torch.svd_lowrank(emb, q=q, niter=2)
        proj = u[:, :d] * s[:d]                    # [vocab, d] == E @ V_d
        del u, s, v
        torch.cuda.empty_cache()
        if not args.no_normalize:
            # LayerNorm after the embedding removes the input-side scale, so the
            # only effect of this constant is the tied head's logit scale. Unit
            # rows keep token-pair cosines and start the frozen head at cosine-
            # scale logits instead of the raw SVD magnitudes (CE 135 at step 0).
            # Unit rows times 1/sqrt(d): the output LayerNorm has norm sqrt(d),
            # so the tied-head logits start at standard-init scale (~1) instead
            # of sqrt(d) * cosine (CE 30 at step 0).
            proj = (torch.nn.functional.normalize(proj, dim=-1)
                    * (d ** -0.5))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proj.to(torch.float16).cpu().contiguous(), out)
    print(f"saved {tuple(proj.shape)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
