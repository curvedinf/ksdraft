"""Verify a ksdraft export: load it back and propose from pure int8 KV.

Usage: python scripts/verify_export.py <export_dir>
"""
import sys

import torch

from ksdraft.export import load_export


def main(path: str) -> int:
    m = load_export(path, device="cpu").eval()
    tier = getattr(m, "_int8_tier", None)
    cfg = m.config
    B, T = 2, 16
    S = int(cfg.kv_window) + T - 1
    taps = int(cfg.layers)
    codes = torch.randint(-128, 127, (B, taps, S, cfg.kv_dim),
                          dtype=torch.int8)
    scales = (torch.rand(B, taps, S, cfg.n_kv_heads, dtype=torch.float32)
              * 0.05 + 1e-4).to(torch.float16)
    toks = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out = m.propose_block_tokens(
            toks, 15, target_kv=(codes, scales, codes, scales))
    assert out.shape == (B, 15), out.shape
    print(f"OK {path}: tier={tier} kv_dim={cfg.kv_dim} "
          f"layers={cfg.layers} proposed={tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
