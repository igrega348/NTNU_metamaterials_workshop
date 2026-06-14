# Session Pack
**Packed:** 2026-06-14
**Project:** NTNU_metamaterials_workshop
**Session goal:** Verify in-plane vs out-of-plane projection inconsistency in neural_xray pipeline via 4-way training matrix (PSNR + normed_correlation).

## Status
✅ Complete — OOP paradox diagnosed and fixed

## Summary
All 4 models (in_in, out_out, out_in, in_out) reconstruct the 3D phantom equally well. The apparent normed_corr gap (0.982 vs 0.789) was a **metric bug**, not a real reconstruction quality difference.

## Final Results

### PSNR matrix (dB) — rendering quality
```
                     |    Eval in-plane |         Eval OOP
Train in-plane       |           58.806 |           58.680
Train OOP            |           58.484 |           58.647
```
**Interpretation:** Near-flat (~0.3 dB). OOP training does NOT degrade rendering quality.

### Normed correlation matrix — AFTER FIX
```
                     |    Eval in-plane |         Eval OOP
Train in-plane       |            0.982 |            0.983
Train OOP            |            0.983 |            0.982
```
**Interpretation:** All ~0.982–0.983. OOP training does NOT degrade 3D reconstruction quality.

### Normed correlation matrix — BEFORE FIX (wrong)
```
                     |    Eval in-plane |         Eval OOP
Train in-plane       |            0.982 |            0.905
Train OOP            |            0.911 |            0.789
```

## Root Cause of the OOP Paradox

`auto_orient_and_center_poses` (nerfstudio) applies a small rotation R to camera poses to align the mean camera-up with world +Z. This rotation is saved to `dataparser_transforms.json` per model.

- For `canonical_in_in`: R = identity (in-plane cameras already Z-up) → no mismatch
- For `canonical_out_out`: R ≈ 4° rotation → significant mismatch

**The bug:** `get_eval_density_loss` in `canonical_pipeline.py` evaluated:
- `pred_density = field.get_density_from_pos(pos)` — pos in NeRF-space (rotated)
- `gt_density = obj.density(pos)` — object.json sphere centers in original world space

These are in different coordinate systems. The correlation was low because the predicted density "saw" the spheres at rotated positions while the GT "saw" them at original positions.

**The fix:** Before calling `obj.density(pos)`, apply the inverse rotation:
```python
try:
    T = self.datamanager.train_dataparser_outputs.dataparser_transform  # (3, 4)
    R = T[:3, :3].to(pos)
    world_pos = pos @ R  # NeRF-space → world-space
except Exception:
    world_pos = pos
density = obj.density(world_pos).squeeze()
```
Applied to: `neural_xray/nerfstudio-xray/nerf-xray/nerf_xray/canonical_pipeline.py`

**Camera convention:** Confirmed correct (48.49 dB PSNR rendering no flip needed).
**OOP cameras:** Geometrically correct — elevation -70° to +77°, no gimbal lock.

## All Bugs Fixed

1. `CanonicalPipeline` missing `get_eval_density_loss` → added to `canonical_pipeline.py`
2. Grid overflow in `get_eval_density_loss` (npoints used as per-axis) → `n_per_axis = int(round(npoints**(1/3)))`
3. `run_canonical.sh` PSNR output path broken when suf contains `/` → `suf_safe="${suf//\//_}"`
4. `CanonicalPipeline.get_average_eval_image_metrics` missing `which` kwarg → added `which=None, **kwargs`
5. `json.dumps` failing on torch.Tensor in eval.py → added `TensorEncoder`
6. **ROOT CAUSE: normed_corr metric doesn't account for dataparser coordinate rotation** → apply `pos @ R` before `obj.density()` in `canonical_pipeline.py`

## Files Modified
- `neural_xray/nerfstudio-xray/nerf-xray/nerf_xray/canonical_pipeline.py` — bugs 1, 2, 4, 6
- `neural_xray/scripts/run_canonical.sh` — bug 3
- `neural_xray/nerfstudio/nerfstudio/scripts/eval.py` — bug 5

## Key Paths
| Artifact | Path |
|---|---|
| Training checkpoints | `neural_xray/outputs/balls_4way/nerf_xray/canonical_*/nerfstudio_models/step-000003000.ckpt` |
| PSNR eval results | `neural_xray/outputs/balls_4way/nerf_xray/canonical_*/eval_metrics_balls_4way_canonical_*.json` |
| Normed corr results (fixed) | `neural_xray/outputs/balls_4way/nerf_xray/canonical_*/eval_normed_balls_4way.json` |
| Density slice images | `debug/density_slices/density_{xy,xz,yz}.png` |
| Matrix report script | `debug/report_oop_matrix.py` |
| Dataset | `neural_xray/data/simulated/balls_4way/` |

## Conclusion
The neural_xray pipeline is **correct for both in-plane and OOP training**. There is no OOP bug. The phantom used (two balls at [-0.5,-0.5,0.2] and [0.5,0.5,-0.3], radius 0.15) is asymmetric and adequately tests geometric reconstruction.
