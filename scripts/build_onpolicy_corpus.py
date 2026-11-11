# SPDX-License-Identifier: Apache-2.0
"""Build the on-policy draft corpus with the deployed Flint target.

The draft must learn the *verifier's* output distribution, not a public
instruction dataset: late-2026 draft training (EAGLE-3.x/4, DFlash2, AMD
Quark) all regenerate responses with the target, apply the exact serving
chat template, and sample at the served temperature. This script does that
with the in-process PTQR INT8 Flint engine.

Tiers (matching the fork's serving traffic):
  - alpaca: instructions from the existing Flint-token corpus, chat-formatted
  - agent:  tool-use legs (read_file / run_tests / grep) with tool schemas
  - doc:    long document spans (summarize / needle retrieval)
  - chat:   short general questions

Each sample is prompt ids (exact serving template, add_generation_prompt=True)
plus the target's sampled response ids; samples are concatenated into a flat
token stream with document `boundaries`, so WindowSampler never crosses one.

  python scripts/build_onpolicy_corpus.py --model <flint dir> \
      --out runs/onpolicy --n-code 2000 --n-agent 200 --n-doc 100 --n-chat 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Serving numerics must match the recipe; set before vllm import.
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "1")
os.environ.setdefault("VLLM_GFX908_ACT_QUANT", "round")
os.environ.setdefault("VLLM_GFX908_INT8_LM_HEAD", "1")
os.environ.setdefault("VLLM_GFX908_MOE_A8W8", "1")
os.environ.setdefault("VLLM_GFX908_O_PROJ_I8", "1")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx908")
os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the test suite or a single test.",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search the workspace with a regex.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                      "path": {"type": "string"}},
                       "required": ["pattern"]}}},
]

SYSTEM = ("You are a precise software engineering agent. Inspect the "
          "workspace with the provided tools and make minimal, tested changes.")

TASKS = [
    "Find where `{sym}` is defined in {path} and explain its call sites.",
    "The test `{sym}` is failing after the last refactor. Locate the cause and fix it.",
    "Add input validation to `{sym}` in {path} and run the tests.",
    "Refactor `{sym}` to remove duplicated branches; keep behaviour identical.",
    "Write a regression test for the bug in {path} and run it.",
]
SYMS = ["parse_config", "session_cache", "router.dispatch", "load_checkpoint",
        "batcher.flush", "metric.record", "auth.refresh", "worker.rescale"]
PATHS = ["src/config.py", "src/server/routes.py", "src/cache/session.py",
         "src/train/loop.py", "src/api/auth.py", "tests/test_router.py"]

QUESTIONS = [
    "Summarize the key points of the document below in a short bullet list.",
    "What are the main risks described in the document? Answer briefly.",
    "Extract the concrete numbers and dates mentioned in the document.",
    "Write a one-paragraph abstract of the document.",
]

NEEDLE_Q = ("The document contains an access code. Find it and reply with the "
            "code only.")


def build_prompts(tok, corpus_path: Path, args, rng: random.Random):
    """Returns a list of (kind, prompt_ids, max_new) tuples."""
    prompts = []
    text = None
    if corpus_path.exists():
        ids = __import__("torch").load(corpus_path, map_location="cpu",
                                       weights_only=True)
        if isinstance(ids, dict):
            ids = ids.get("tokens", ids.get("ids"))
        text = tok.decode(ids.tolist())

    # tier 1: instructions mined from the Flint-token corpus
    instrs = []
    if text:
        for part in text.split("### Instruction:")[1:]:
            ins = part.split("### Response:")[0].strip()
            if 40 <= len(ins) <= 4000:
                instrs.append(ins)
    rng.shuffle(instrs)
    for ins in instrs[:args.n_code]:
        msgs = [{"role": "user", "content": ins}]
        prompts.append(("alpaca", msgs, None, args.max_new))

    # tier 2: agentic tool-use legs
    for _ in range(args.n_agent):
        task = rng.choice(TASKS).format(sym=rng.choice(SYMS),
                                        path=rng.choice(PATHS))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
        prompts.append(("agent", msgs, TOOLS, args.max_new))

    # tier 3: long document spans (summaries + needle retrieval)
    if text and args.n_doc:
        body = text[len(text) // 3: 2 * len(text) // 3]
        span_chars = max(2000, args.doc_chars)
        spans = [body[i:i + span_chars]
                 for i in range(0, len(body) - span_chars,
                                span_chars * 3 // 2)]
        rng.shuffle(spans)
        for i, span in enumerate(spans[:args.n_doc]):
            if i % 2 == 0:
                q = rng.choice(QUESTIONS)
                msgs = [{"role": "user",
                         "content": f"{q}\n\n---\n{span}"}]
            else:
                code = f"ZK{rng.randrange(10**8):08d}"
                cut = rng.randrange(len(span) // 4, 3 * len(span) // 4)
                doc = (span[:cut] +
                       f"\nThe access code for PROJECT-{rng.randrange(10**6):06d} "
                       f"is {code}.\n" + span[cut:])
                msgs = [{"role": "user",
                         "content": f"{NEEDLE_Q}\n\n---\n{doc}"}]
            prompts.append(("doc", msgs, None, args.doc_new))

    # tier 4: short general chat
    for _ in range(args.n_chat):
        q = rng.choice([
            "Explain how a hash table handles collisions.",
            "What is the difference between a process and a thread?",
            "Give me a short checklist for reviewing a database migration.",
            "Why does floating point addition lose associativity?",
            "Summarize the CAP theorem in three sentences.",
        ])
        prompts.append(("chat", [{"role": "user", "content": q}], None, 256))

    out = []
    for kind, msgs, tools, max_new in prompts:
        try:
            text = tok.apply_chat_template(
                msgs, tools=tools, tokenize=False,
                add_generation_prompt=True)
        except Exception as exc:
            print(f"[skip] {kind}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not text.endswith("<|assistant|>"):
            continue
        out.append({"kind": kind, "prompt": text, "max_new": max_new})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="tokenizer/model dir (same one the server serves)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8030/v1")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--corpus", default="/home/curved/SDGraft/data/flint_longctx/train_tokens.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-code", type=int, default=2000)
    ap.add_argument("--n-agent", type=int, default=200)
    ap.add_argument("--n-doc", type=int, default=100)
    ap.add_argument("--n-chat", type=int, default=400)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--doc-chars", type=int, default=8000)
    ap.add_argument("--doc-new", type=int, default=256)
    ap.add_argument("--max-prompt", type=int, default=12000)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--greedy-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    import json as _json
    import concurrent.futures as cf
    import urllib.request

    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = build_prompts(tok, Path(args.corpus), args, rng)
    if not prompts:
        raise SystemExit("no prompts built")
    print(f"prompts: {len(prompts)}", flush=True)

    def one(item, idx):
        body = {"model": "flint", "prompt": item["prompt"],
                "max_tokens": item["max_new"]}
        if idx % max(1, int(1 / max(args.greedy_frac, 1e-6))) == 0:
            body["temperature"] = 0.0
        else:
            body["temperature"] = 1.0
            body["top_p"] = 0.95
            body["top_k"] = 20
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/completions",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {args.api_key}"}
                        if args.api_key else {})})
        with urllib.request.urlopen(req, timeout=900) as resp:
            out = _json.loads(resp.read())
        return out["choices"][0]["text"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens, bounds, meta = [], [], []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(one, p, i): (i, p)
                for i, p in enumerate(prompts)}
        for done, fut in enumerate(cf.as_completed(futs), 1):
            i, item = futs[fut]
            try:
                text = fut.result()
            except Exception as exc:
                print(f"[fail] {i}: {type(exc).__name__}: {exc}", flush=True)
                continue
            prompt_ids = tok.encode(item["prompt"], add_special_tokens=False)
            ids = tok.encode(item["prompt"] + (text or ""),
                             add_special_tokens=False)
            if len(ids) < 8:
                continue
            bounds.append(len(tokens))
            tokens.extend(ids)
            meta.append({"kind": item["kind"],
                         "prompt": len(prompt_ids),
                         "response": max(0, len(ids) - len(prompt_ids))})
            if done % 64 == 0:
                print(f"[{done}/{len(prompts)}] tokens {len(tokens)} "
                      f"{time.time() - t0:.0f}s", flush=True)
                torch_save(out_dir / f"shard_{done:06d}.pt", tokens, bounds,
                           meta)
    torch_save(out_dir / f"shard_{len(meta):06d}.pt", tokens, bounds, meta)
    summary = {"model": args.model, "samples": len(meta),
               "prompt_tokens": sum(m["prompt"] for m in meta),
               "response_tokens": sum(m["response"] for m in meta),
               "temp": 1.0, "top_p": 0.95, "top_k": 20,
               "greedy_frac": args.greedy_frac, "seed": args.seed,
               "kinds": {k: sum(1 for m in meta if m["kind"] == k)
                         for k in {m["kind"] for m in meta}}}
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 0


def torch_save(path: Path, tokens, bounds, meta) -> None:
    import torch
    torch.save({"tokens": torch.tensor(tokens, dtype=torch.int64),
                "boundaries": torch.tensor(bounds, dtype=torch.int64),
                "meta": meta[-4096:]}, path)
    print(f"wrote {path} ({len(tokens)} tokens, {len(bounds)} docs)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
