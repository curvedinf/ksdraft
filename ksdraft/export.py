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
                 dtype: str = "fp16", share_vocab: bool = False) -> Path:
    """Write a portable draft checkpoint.

    ``share_vocab=True`` (the DFly2 pattern): the artifact omits the tied
    vocab table and records ``shared_embedding`` — the serving side binds
    the target's own embedding/lm_head, so the draft ships only its trunk.
    """
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
    if share_vocab:
        meta["shared_embedding"] = True
        if not c.tie_word_embeddings:
            # the target holds both tables; the draft artifact is trunk-only
            meta["shared_head"] = True
    if getattr(model, "spine_core", None) is not None:
        meta["transfer_rank"] = int(model.spine_core.rank)
    if getattr(model, "teacher_dim", 0):
        meta["teacher_dim"] = model.teacher_dim
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    if dtype == "int8":
        from .kernels.ks3_int8 import KS3Int8Cell
        from .kernels.linear import Int8Linear
        from .model import to_int8

        model = to_int8(model)
        sd = {}
        skip = set()
        for name, mod in model.named_modules():
            if isinstance(mod, KS3Int8Cell) and mod.trainable:
                mod._repack_masters()
                skip.update(f"{name}.{key}" for key, _ in mod.named_parameters())
            if isinstance(mod, Int8Linear):
                mod._quantize_weight()
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
    if share_vocab:
        # the target already holds the table(s); the draft artifact is trunk-only
        sd.pop("embedding.weight", None)
        if not c.tie_word_embeddings:
            sd.pop("head.weight", None)
        # the DSpine transfer table is target-derived support for the proposal
        # walk, not a drafter weight; it does not ship
        sd.pop("transfer_C", None)
    try:
        from safetensors.torch import save_file
        save_file(sd, str(out / "ksdraft.safetensors"),
                  metadata={"format": "ksdraft-v1"})
        weights = out / "ksdraft.safetensors"
    except ImportError:
        weights = out / "ksdraft.pt"
        torch.save(sd, weights)
    # HF-repo pattern (RESEARCH Section 12): the checkpoint directory carries
    # its own loader, so engines consume it without guessing.
    (out / "modeling_ksdraft.py").write_text(LOADER_SHIM)
    return weights


LOADER_SHIM = '# SPDX-License-Identifier: Apache-2.0\n"""Loader for a ksdraft draft repo (checkpoint + config + this shim).\n\n    from modeling_ksdraft import from_pretrained\n    draft = from_pretrained("path/to/repo", device="cuda")\n\nThe model class is the package\'s ksdraft.model.KSDraftModel; proposals\ngo through propose_block_tokens / block_logits (see ksdraft.export).\nShared-embedding artifacts require the target\'s [vocab, d] tables —\nboth, when the draft is untied:\n\n    draft = from_pretrained("path/to/repo", embedding=target_embed,\n                             head=target_lm_head)\n"""\nfrom ksdraft.export import load_export\n\n\ndef from_pretrained(path, device="cpu", embedding=None, head=None):\n    return load_export(str(path), device=device, embedding=embedding,\n                       head=head)\n\n\ndef load_draft(path, device="cpu", embedding=None, head=None):\n    return load_export(str(path), device=device, embedding=embedding,\n                       head=head)\n'


def load_export(out_dir: str, device: str = "cpu",
                embedding: torch.Tensor | None = None,
                head: torch.Tensor | None = None) -> KSDraftModel:
    """Rebuild an exported draft.

    ``embedding`` / ``head``: the external [vocab, d_model] tables a
    shared-embedding artifact expects (the target's own tables — dims must
    match exactly). A tied artifact takes one; an untied (DFly2-pattern)
    artifact takes both.
    """
    out = Path(out_dir)
    meta = json.loads((out / "config.json").read_text())
    s = meta["config"]
    cfg = KSDraftConfig(
        vocab_size=s["vocab_size"], layers=s["layers"], d_model=s["d_model"],
        head_size=s["head_size"], rosa_streams=s["rosa_streams"],
        rosa_bits=s["rosa_bits"], ks3_charts=s["ks3_charts"],
        block_len=s.get("block_len", 16),
        dtype=s["dtype"],
        tie_word_embeddings=s.get("tie_word_embeddings", True),
        kv_dim=s.get("kv_dim", 0), n_kv_heads=s.get("n_kv_heads", 8),
        kv_window=s.get("kv_window", 1024),
    )
    shared = bool(meta.get("shared_embedding"))
    if shared:
        if embedding is None:
            raise ValueError(
                "this artifact shares the target's embedding (shared_embedding): "
                "pass the target's [vocab, d_model] table via embedding=")
        if not cfg.tie_word_embeddings and head is None:
            raise ValueError(
                "this artifact is untied and shares both tables "
                "(shared_head): pass the target's lm_head via head=")
    for name, tensor in (("embedding", embedding), ("head", head)):
        if tensor is not None and tuple(tensor.shape) != (
                cfg.vocab_size, cfg.d_model):
            raise ValueError(
                f"external {name} {tuple(tensor.shape)} does not match "
                f"the draft's [{cfg.vocab_size}, {cfg.d_model}]")
    ts_space = None
    if meta.get("transfer_rank"):
        ts_space = {"rank": int(meta["transfer_rank"])}
    model = KSDraftModel(cfg, teacher_dim=meta.get("teacher_dim", 0),
                         transfer_space=ts_space)
    if shared:
        with torch.no_grad():
            # in-place copy keeps the tie when tied: head.weight is embedding.weight
            model.embedding.weight.copy_(
                embedding.to(model.embedding.weight.dtype))
            if not cfg.tie_word_embeddings:
                model.head.weight.copy_(
                    head.to(model.head.weight.dtype))
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
    if shared:
        # the external table(s) were copied in above, not loaded from state
        missing = [k for k in missing
                   if k not in ("embedding.weight", "head.weight")]
    if missing:
        print(f"[load_export] {len(missing)} new parameter(s) fresh: "
              f"{missing[:3]}")
    return model.to(device)
