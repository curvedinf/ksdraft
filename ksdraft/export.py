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


def export_model(model: KSDraftModel, out_dir: str,
                 dtype: str = "fp16") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = model.config
    meta = {
        "format": "ksdraft-v1",
        "package_version": __version__,
        # HF-style top-level fields: the serving engine's ModelConfig parses
        # these directly (draft_model_config for the ksdraft speculator).
        "model_type": "ksdraft",
        "architectures": ["KSDraftModelForCausalLM"],
        "hidden_size": c.d_model,
        "num_hidden_layers": c.layers,
        "vocab_size": c.vocab_size,
        "torch_dtype": "float32",
        "max_position_embeddings": 4096,
        "dtype_tiers": ["int8", "bf16", "fp32"],
        "dtype": dtype,
        "config": c.summary(),
    }
    if getattr(model, "spine_core", None) is not None:
        meta["transfer_rank"] = int(model.spine_core.rank)
    if getattr(model, "teacher_dim", 0):
        meta["teacher_dim"] = model.teacher_dim
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    if dtype == "int8":
        from .kernels.linear import Int8Linear
        from .model import to_int8

        model = to_int8(model)
        sd = {}
        skip = set()
        for name, mod in model.named_modules():
            if isinstance(mod, Int8Linear):
                sd[f"{name}.weight_codes"] = mod.weight_codes.detach().cpu().contiguous()
                sd[f"{name}.weight_scales"] = mod.weight_scales.detach().cpu().float()
                skip.add(f"{name}.weight")
        for k, v in model.state_dict().items():
            if k in skip:
                continue
            dt = torch.float16 if v.is_floating_point() else v.dtype
            sd[k] = v.detach().cpu().to(dt).contiguous()
    else:
        sd = {k: v.detach().cpu().contiguous()
              for k, v in model.state_dict().items()}
        if dtype in ("fp16", "bf16"):
            half = torch.float16 if dtype == "fp16" else torch.bfloat16
            sd = {k: (v.to(half) if v.is_floating_point() else v)
                  for k, v in sd.items()}
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
        block_len=s.get("block_len", 16),
        dtype=s["dtype"], tie_word_embeddings=True,
    )
    model = KSDraftModel(cfg, teacher_dim=meta.get("teacher_dim", 0))
    st_path = out / "ksdraft.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file
        sd = dict(load_file(str(st_path)))
    else:
        sd = dict(torch.load(out / "ksdraft.pt", map_location="cpu"))

    if meta.get("dtype") == "int8":
        from .kernels.quant import dequantize_last_dim
        from .model import to_int8

        model = to_int8(model)
        # Rebuild masters from the stored codes+scales: dither=0 quantization
        # of the dequantized masters reproduces the exported codes exactly.
        for k in [k for k in sd if k.endswith(".weight_codes")]:
            path = k[: -len(".weight_codes")]
            codes = sd.pop(k)
            scales = sd.pop(path + ".weight_scales")
            sd[path + ".weight"] = dequantize_last_dim(
                codes.float(), scales.float())
    if "head.weight" not in sd and "embedding.weight" in sd:
        sd["head.weight"] = sd["embedding.weight"]  # restore the tie

    # Exports made before the parallel block head (or other new surfaces)
    # load with those parameters freshly initialized.
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_export] {len(missing)} new parameter(s) fresh: "
              f"{missing[:3]}")
    return model.to(device)
