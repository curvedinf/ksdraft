"""Gate battery (RESEARCH.md Sections 8.1/11 Stage 7, Section 12A `gate`).

Checks, on a fresh model of the configured shape (or a loaded checkpoint):

1. parameter budget matches the config formulas exactly
2. init doctrine: every fresh block is exactly identity
3. bf16 tier parity vs fp32 (the fallback tier must track the reference)
4. INT8 KS3 cell parity vs the torch KS3 per nonlinear mode
5. credit-engine availability (informational — the Rosa training path)
6. forward finiteness end-to-end

Writes a JSON report; returns an overall pass/fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import KSDraftConfig
from .model import KSDraftModel


def run_gate(config: KSDraftConfig | None = None, device: str = "cpu",
             model: KSDraftModel | None = None, out_dir: str | None = None) -> dict:
    cfg = config or model.config if model is not None else KSDraftConfig()
    if model is None:
        torch.manual_seed(0)
        model = KSDraftModel(cfg).to(device)
    model.eval()
    checks: dict = {}

    # 1. parameter budget
    n = sum(p.numel() for p in model.parameters())
    counts = model.param_counts()
    expected = cfg.total_params() + counts["extras"]
    checks["param_budget"] = {
        "pass": n == expected,
        "detail": f"{n:,} params vs formula {expected:,} "
                  f"(core {counts['core']:,} + extras {counts['extras']:,})",
    }

    # 2. identity-block init
    fresh = KSDraftModel(cfg, device=device).eval()
    x = torch.randn(2, 8, cfg.d_model, device=device)
    with torch.no_grad():
        y, _ = fresh.blocks[0](x.clone())
    ok = torch.allclose(y, x, atol=1e-5)
    checks["identity_block_init"] = {"pass": bool(ok), "detail": f"max dev {(y - x).abs().max().item():.2e}"}

    # 3. bf16 tier parity vs fp32
    tokens = torch.randint(0, min(cfg.vocab_size, 1000), (2, 16), device=device)
    with torch.no_grad():
        y32, _ = model(tokens)
        ybf, _ = model.to(torch.bfloat16)(tokens)
    rel = ((ybf.float() - y32.float()).norm() / y32.float().norm()).item()
    checks["bf16_parity"] = {"pass": rel < 0.05, "detail": f"rel {rel:.4f} (threshold 0.05)"}

    # 4. INT8 KS3 cell parity per nonlinear mode
    from .model.ks3 import KS3
    from .kernels.ks3_int8 import KS3Int8Cell
    torch.manual_seed(1)
    cell = KS3(min(cfg.d_model, 128), 256).to(device)
    with torch.no_grad():
        cell.up.weight.normal_(0, 0.05)
    xin = torch.randn(32, cell.d_model, device=device)
    with torch.no_grad():
        ref = cell(xin)
    mode_results = {}
    for mode, tol in (("fp32", 0.05), ("fp16", 0.05), ("int8", 0.08)):
        q = KS3Int8Cell(cell, group=128, nonlinear=mode).to(device).eval()
        with torch.no_grad():
            relm = ((q(xin) - ref).norm() / ref.norm()).item()
        mode_results[mode] = {"rel": round(relm, 4), "pass": relm < tol}
    checks["int8_ks3_parities"] = {
        "pass": all(m["pass"] for m in mode_results.values()),
        "detail": mode_results,
    }

    # 5. credit engine availability (informational)
    try:
        from .kernels.rosa_credit import engine_available
        avail = engine_available()
    except Exception:
        avail = False
    checks["rosa_credit_engine"] = {
        "pass": True,  # informational: eval-only torch path is always valid
        "detail": f"available={avail} (training-mode Rosa uses counterfactual "
                  f"credits when present; otherwise payload/out_proj only)",
    }

    # 6. end-to-end finiteness
    with torch.no_grad():
        logits, feats = model(tokens)
    finite = torch.isfinite(logits).all() and all(torch.isfinite(f).all() for f in feats)
    checks["forward_finite"] = {"pass": bool(finite), "detail": f"logits {tuple(logits.shape)}"}

    report = {
        "config": cfg.summary(),
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks.values()),
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate_report.json").write_text(json.dumps(report, indent=2))
    return report
