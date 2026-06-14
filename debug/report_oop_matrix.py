#!/usr/bin/env python3
"""Phase 4: collect PSNR and normed_correlation from all 4 OOP models and print the 2x2 matrix."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = REPO_ROOT / "neural_xray" / "outputs" / "balls_4way"

MODELS = {
    "in_in":  "nerf_xray/canonical_in_in",
    "out_out": "nerf_xray/canonical_out_out",
    "in_out": "nerf_xray/canonical_in_out",
    "out_in": "nerf_xray/canonical_out_in",
}

def load_psnr(model_dir: Path) -> float | None:
    for fn in model_dir.glob("eval_metrics_*.json"):
        d = json.loads(fn.read_text())
        r = d.get("results", {})
        for key in ("psnr", "PSNR"):
            if key in r:
                v = r[key]
                return v["mean"] if isinstance(v, dict) else float(v)
    return None

def load_normed_corr(model_dir: Path) -> float | None:
    for fn in model_dir.glob("eval_normed_*.json"):
        d = json.loads(fn.read_text())
        r = d.get("results", {})
        # ComputeNormedCorrelation stores results keyed by time
        for t_key, metrics in r.items():
            for key in ("normed_correlation", "correlation", "normed_corr"):
                if key in metrics:
                    v = metrics[key]
                    return v["mean"] if isinstance(v, dict) else float(v)
    return None

def fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"

print("\n=== OOP Verification Matrix ===\n")

# Collect all values
vals = {}
for name, suf in MODELS.items():
    d = OUTDIR / suf
    psnr = load_psnr(d)
    nc = load_normed_corr(d)
    vals[name] = (psnr, nc)
    status = "OK" if d.exists() else "MISSING"
    print(f"  {name:8s} ({status}): PSNR={fmt(psnr)}, normed_corr={fmt(nc)}")

print()
print("--- PSNR matrix (dB) ---")
print(f"{'':20s} | {'Eval in-plane':>16s} | {'Eval OOP':>16s}")
print(f"{'Train in-plane':20s} | {fmt(vals['in_in'][0]):>16s} | {fmt(vals['in_out'][0]):>16s}")
print(f"{'Train OOP':20s} | {fmt(vals['out_in'][0]):>16s} | {fmt(vals['out_out'][0]):>16s}")

print()
print("--- Normed correlation matrix ---")
print(f"{'':20s} | {'Eval in-plane':>16s} | {'Eval OOP':>16s}")
print(f"{'Train in-plane':20s} | {fmt(vals['in_in'][1]):>16s} | {fmt(vals['in_out'][1]):>16s}")
print(f"{'Train OOP':20s} | {fmt(vals['out_in'][1]):>16s} | {fmt(vals['out_out'][1]):>16s}")

print()
print("--- Interpretation ---")
ii, oo = vals['in_in'][0], vals['out_out'][0]
io, oi = vals['in_out'][0], vals['out_in'][0]
if None not in (ii, oo, io, oi):
    if oo is not None and ii is not None and (ii - oo) > 3:
        print("  WARNING: out_out << in_in → OOP camera matrices likely wrong (gimbal-lock or matrix bug)")
    elif oi is not None and ii is not None and abs(oi - ii) < 1:
        print("  WARNING: out_in ≈ in_in → OOP flag may not produce distinct views")
    elif io is not None and ii is not None and io > ii:
        print("  WARNING: in_out > in_in → eval-side coordinate confusion")
    else:
        print("  Pipeline looks healthy: in_in ≈ out_out, cross-conditions lower")
else:
    print("  (Some results not yet available)")
