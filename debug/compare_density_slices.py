"""Compare density slices from canonical_in_in vs canonical_out_out reconstructions."""
import sys
import json
from pathlib import Path

import numpy as np
import cv2 as cv
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "neural_xray/nerfstudio-xray/nerf-xray"))
sys.path.insert(0, str(ROOT / "neural_xray/nerfstudio"))

from nerfstudio.scripts.eval import eval_setup

OUTDIR = ROOT / "debug/density_slices"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "in_in":  ROOT / "neural_xray/outputs/balls_4way/nerf_xray/canonical_in_in/config.yml",
    "out_out": ROOT / "neural_xray/outputs/balls_4way/nerf_xray/canonical_out_out/config.yml",
    "out_in": ROOT / "neural_xray/outputs/balls_4way/nerf_xray/canonical_out_in/config.yml",
    "in_out": ROOT / "neural_xray/outputs/balls_4way/nerf_xray/canonical_in_out/config.yml",
}

PLANES = ["xy", "xz", "yz"]
DISTANCES = [0.0]  # centre slice
RESOLUTION = 400

slices = {}  # model -> plane -> numpy array

for name, cfg in MODELS.items():
    print(f"\n=== Loading {name} ===")
    _, pipeline, ckpt, _ = eval_setup(cfg)
    pipeline.eval()

    slices[name] = {}
    with torch.no_grad():
        # GT (same for all models, but export once per model to confirm alignment)
        for plane in PLANES:
            gt   = pipeline.eval_along_plane("datamanager", plane=plane, distance=0.0,
                                             resolution=RESOLUTION, engine="numpy")
            pred = pipeline.eval_along_plane("field",       plane=plane, distance=0.0,
                                             resolution=RESOLUTION, engine="numpy", rhomax=float(gt.max()) if gt.max()>0 else 1.0)
            slices[name][plane] = {"gt": gt, "pred": pred}
            print(f"  {plane}: gt max={gt.max():.4f}  pred max={pred.max():.4f}")

    del pipeline
    torch.cuda.empty_cache()

# Build comparison images: for each plane, stack [GT | in_in | out_out | out_in | in_out]
def to_uint8(arr):
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)

gt_saved = False
summary = {}
for plane in PLANES:
    rows = []
    labels = []

    # GT (from in_in pipeline, same object)
    gt = slices["in_in"][plane]["gt"]
    rows.append(to_uint8(gt))
    labels.append("GT")

    for name in MODELS:
        pred = slices[name][plane]["pred"]
        rows.append(to_uint8(pred))
        labels.append(name)

    # Stack horizontally with 2px white separator
    sep = np.full((RESOLUTION, 2), 255, dtype=np.uint8)
    composite = rows[0]
    for r in rows[1:]:
        composite = np.hstack([composite, sep, r])

    # Add label bar (white strip at bottom)
    label_h = 20
    label_bar = np.full((label_h, composite.shape[1]), 240, dtype=np.uint8)
    x = 0
    cell_w = RESOLUTION + 2
    for i, lbl in enumerate(labels):
        cx = x + RESOLUTION // 2
        cv.putText(label_bar, lbl, (cx - 30, 14), cv.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        x += cell_w

    composite = np.vstack([composite, label_bar])

    fn = OUTDIR / f"density_{plane}.png"
    cv.imwrite(str(fn), composite)
    print(f"Saved {fn}")

    # Compute per-model stats for summary
    for name in MODELS:
        pred = slices[name][plane]["pred"]
        gt_a = slices[name][plane]["gt"]
        gt_n = gt_a / (gt_a.max() + 1e-8)
        pr_n = pred / (pred.max() + 1e-8)
        mse = float(np.mean((gt_n - pr_n) ** 2))
        summary.setdefault(name, {})[plane] = {"slice_mse": mse}

print("\n=== Slice MSE (normalised) ===")
for name, planes in summary.items():
    for plane, m in planes.items():
        print(f"  {name}  {plane}: {m['slice_mse']:.6f}")

(OUTDIR / "slice_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nDone. Images at {OUTDIR}")
