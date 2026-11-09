"""Export (RESEARCH.md Section 12): portable checkpoint + config.

Writes `<out>/config.json` (shape summary, dtype tier, package version) and
weights as safetensors when available (engine-portable) else a torch .pt
state dict. One checkpoint serves every dtype tier — masters are stored and
INT8 packs derive at load (Section 12B).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import __version__
from .config import KSDraftConfig
from .model import KSDraftModel


def export_model(model: KSDraftModel, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "ksdraft-v1",
        "package_version": __version__,
        "model_type": "ksdraft",
        "dtype_tiers": ["int8", "bf16", "fp32"],
        "architectures": ["KSDraftModelForCausalLM"],
        "config": model.config.summary(),
    }
    if getattr(model, "teacher_dim", 0):
        meta["teacher_dim"] = model.teacher_dim
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if model.head.weight is model.embedding.weight:
        sd.pop("head.weight", None)  # safetensors forbids shared storage; retied on load
    try:
        from safetensors.torch import save_file
        save_file(sd, str(out / "ksdraft.safetensors"),
                  metadata={"format": "ksdraft-v1"})
        weights = out / "ksdraft.safetensors"
    except ImportError:
        weights = out / "ksdraft.pt"
        torch.save(sd, weights)
    return weights


def load_export(out_dir: str, device: str = "cpu") -> KSDraftModel:
    out = Path(out_dir)
    meta = json.loads((out / "config.json").read_text())
    s = meta["config"]
    cfg = KSDraftConfig(
        vocab_size=s["vocab_size"], layers=s["layers"], d_model=s["d_model"],
        head_size=s["head_size"], rosa_streams=s["rosa_streams"],
        rosa_bits=s["rosa_bits"], ks3_charts=s["ks3_charts"],
        dtype=s["dtype"], tie_word_embeddings=True,
    )
    model = KSDraftModel(cfg, teacher_dim=meta.get("teacher_dim", 0))
    st_path = out / "ksdraft.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file
        sd = load_file(str(st_path))
        if "head.weight" not in sd:
            sd["head.weight"] = sd["embedding.weight"]  # restore the tie
    else:
        sd = torch.load(out / "ksdraft.pt", map_location=device)
    model.load_state_dict(sd)
    return model.to(device)
