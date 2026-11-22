# SPDX-License-Identifier: Apache-2.0
"""Mine the Inkling target's signals over a tokenized corpus.

Runs the BF16-decoded teacher (transformers fork, layers spread across the
four MI100s) chunk by chunk and stores, per token position:

- k_stream / v_stream: the 4 tap layers' cached K/V (8 heads x 128 = 1024)
  pushed through the deployed int8_block_g128 grid (the fork's cache-write
  kernel semantics, via ksdraft.teacher.emulate_int8_kv), fp16;
- topk_ids / topk_probs: the teacher's top-32 next-token distribution at
  T=1.0 with NO top-p/top-k truncation (the verified official contract).

Taps are SLIDING-WINDOW layers only, so the mined KV equals the deployed
cache content exactly within W=1024 (no truncation loss):

    layer 4  (prelude, SWA)          layer 10 (loop core, final pass, SWA)
    layer 16 (loop core, final pass, SWA)  layer 22 (coda, SWA)

Looped layers fire 3x per forward; the hook keeps the LAST firing.

  PYTHONPATH=/home/curved/models/inkling-transformers-src/src \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  ~/quant-venv/bin/python scripts/mine_inkling_signals.py \
      --source ~/models/Inkling-2-Mini-source \
      --corpus runs/inkling_corpus/base \
      --out runs/inkling_signals --tokens 8000000 --report-every 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

KSROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KSROOT))  # ksdraft.teacher.emulate_int8_kv

TAPS = [4, 10, 16, 22]
LOOP_CORE = (6, 18)  # layers 6..17 loop 3x
TOPK = 32


def place_teacher_inputs(model):
    """Run each physical layer on its weight device, residuals on GPU 0.

    Same treatment as the fork's inkling2_teacher.place_teacher_inputs:
    accelerate's nested offload hooks otherwise move operands back to the
    root device inside a remote layer (cross-device kernels fail with
    hipErrorInvalidValue without this).
    """
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import send_to_device

    remove_hook_from_module(model, recurse=True)
    root = model.get_input_embeddings().weight.device
    for layer in model.model.language_model.layers:
        device = next(layer.parameters()).device
        if device == root:
            continue

        def enter(module, args, kwargs, *, dest=device):
            return send_to_device(args, dest), send_to_device(kwargs, dest)

        def leave(module, args, output):
            return send_to_device(output, root)

        layer.register_forward_pre_hook(enter, with_kwargs=True)
        layer.register_forward_hook(leave)


def load_teacher(source: Path):
    import re

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForConditionalGeneration)

    config = AutoConfig.from_pretrained(str(source), local_files_only=True)
    # The HF format converter dequantizes the NVFP4/MXFP8 experts to BF16
    # (same route as the fork's inkling2_teacher.py).
    del config.quantization_config
    devices = torch.cuda.device_count()
    if devices < 2:
        raise SystemExit("the decoded teacher needs multiple MI100s")
    layers = config.text_config.num_hidden_layers
    ranks = [i * devices // layers for i in range(layers)]
    # device targets must be real torch.device objects — this transformers
    # build's loader passes them straight to Tensor.to(), where bare ints
    # fail with hipErrorInvalidValue
    gpu = lambda r: torch.device("cuda", r)
    device_map = {"model.vision_tower": gpu(0), "lm_head": gpu(0)}
    for name in ("embed_tokens", "final_input", "norm", "message_input"):
        device_map[f"model.language_model.{name}"] = gpu(0)
    device_map.update({
        f"model.language_model.layers.{i}": gpu(rank)
        for i, rank in enumerate(ranks)})

    shard = next(source.glob("model-*.safetensors"))
    weights = {}
    with safe_open(str(shard), framework="pt") as f:
        for name in f.keys():
            if not name.startswith(("model.llm.", "model.visual.")):
                raise ValueError(f"unexpected tensor: {name}")
            m = re.match(r"model\.llm\.layers\.(\d+)\.", name)
            rank = ranks[int(m.group(1))] if m else 0
            weights[name] = f.get_tensor(name).to(torch.device("cuda", rank))
    print(f"loaded {len(weights):,} tensors ({devices} GPUs)", flush=True)
    teacher, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        None, state_dict=weights, config=config, device_map=device_map,
        dtype=torch.bfloat16, attn_implementation="eager",
        experts_implementation="eager", local_files_only=True,
        output_loading_info=True)
    errors = {k: v for k, v in info.items() if v}
    if errors:
        raise SystemExit(f"incomplete teacher load: {errors}")
    place_teacher_inputs(teacher)
    teacher.eval().requires_grad_(False)
    return teacher


class TapHooks:
    """Capture final-pass K/V of the tap layers for each chunk."""

    def __init__(self, model):
        self.captured = {}
        self._counts = {}
        lm = model.model.language_model
        for layer in TAPS:
            attn = lm.layers[layer].self_attn
            for name in ("k_proj", "v_proj"):
                key = (layer, name)
                self._counts[key] = 0
                attn.__getattr__(name).register_forward_hook(
                    self._make(key))

    def _make(self, key):
        fires = 3 if LOOP_CORE[0] <= key[0] < LOOP_CORE[1] else 1

        def hook(_m, _i, out):
            self._counts[key] += 1
            if self._counts[key] == fires:
                t = out[0] if isinstance(out, tuple) else out
                # (batch, num_kv_heads, seq, head_dim) -> (seq, 1024)
                b, h, s, d = t.shape
                self.captured[key] = t[0].transpose(0, 1).reshape(s, h * d)
        return hook

    def reset(self):
        self.captured.clear()
        for k in self._counts:
            self._counts[k] = 0


def main() -> int:
    global TAPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/home/curved/models/Inkling-2-Mini-source")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=8_000_000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--taps", type=int, nargs="*", default=TAPS)
    args = ap.parse_args()

    TAPS = list(args.taps)

    from ksdraft.teacher import emulate_int8_kv

    corpus = Path(args.corpus)
    tokens = torch.load(corpus / "tokens.pt", weights_only=True).long()
    bounds = torch.load(corpus / "boundaries.pt", weights_only=True).long()
    total = min(args.tokens, tokens.numel())
    tokens = tokens[:total]
    n_chunks = (total + args.chunk - 1) // args.chunk

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = len(TAPS)
    # Streams are ~16KB/token (4 taps x 1024 x 2B x K/V): far beyond RAM at
    # multi-million-token scale, so rows stream to disk per chunk instead of
    # materializing. topk tensors stay in RAM (6B/token).
    k_f = open(out / "k_stream.bin", "wb")
    v_f = open(out / "v_stream.bin", "wb")
    ti_mm = torch.empty((total, TOPK), dtype=torch.int32)
    tp_mm = torch.empty((total, TOPK), dtype=torch.float16)

    model = load_teacher(Path(args.source))
    hooks = TapHooks(model)

    t0 = time.time()
    with torch.no_grad():
        for c in range(n_chunks):
            lo = c * args.chunk
            hi = min(lo + args.chunk, total)
            ids = tokens[lo:hi].unsqueeze(0).cuda(0)
            hooks.reset()
            logits = model(input_ids=ids).logits[0].float()  # (hi-lo, V)  # (hi-lo, V)
            probs = torch.softmax(logits, dim=-1)
            tp, ti = probs.topk(TOPK, dim=-1)
            ti_mm[lo:hi] = ti.cpu()
            tp_mm[lo:hi] = tp.half().cpu()
            k_rows = torch.empty((hi - lo, L, 1024), dtype=torch.float16)
            v_rows = torch.empty((hi - lo, L, 1024), dtype=torch.float16)
            for i, layer in enumerate(TAPS):
                for j, name in enumerate(("k", "v")):
                    t = hooks.captured[(layer, f"{name}_proj")]
                    grid = emulate_int8_kv(t.float().cpu())
                    (k_rows if name == "k" else v_rows)[
                        :, i] = grid.half()
            k_f.write(k_rows.numpy().tobytes())
            v_f.write(v_rows.numpy().tobytes())
            if c % args.report_every == 0 or c == n_chunks - 1:
                done = hi
                rate = done / max(1e-9, time.time() - t0)
                print(f"chunk {c + 1}/{n_chunks}: {done:,}/{total:,} tokens "
                      f"({rate:,.0f} tok/s)", flush=True)

    k_f.close()
    v_f.close()
    torch.save(ti_mm, out / "topk_ids.pt")
    torch.save(tp_mm, out / "topk_probs.pt")
    b = bounds[bounds <= total]
    torch.save(b, out / "boundaries.pt")
    (out / "meta.json").write_text(json.dumps({
        "source": str(args.source), "corpus": str(corpus),
        "tokens": int(total), "taps": TAPS, "topk": TOPK,
        "kv_grid": "int8_block_g128 emulation (fork kernel semantics)",
        "elapsed_s": round(time.time() - t0, 1),
        "stream_dtype": "fp16", "kv_bytes_per_token": L * 1024 * 4}, indent=2))
    print(f"saved {total:,} positions of 4-tap KV + top-{TOPK} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
