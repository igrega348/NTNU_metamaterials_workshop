# Session Pack
**Packed:** 2026-06-21
**Project:** NTNU_metamaterials_workshop
**Session goal:** Complete 5-stage NeRF training pipeline, build wheel CI matrix, and create cross-section evaluation figures.

## Status
✅ Objective complete

## Completed

- **Full 5-stage training pipeline complete** — all checkpoints verified:
  - canonical_F: `outputs/kelvin/nerf_xray/canonical_F/nerfstudio_models/step-000002000.ckpt`
  - canonical_B: `outputs/kelvin/nerf_xray/canonical_B/nerfstudio_models/step-000002000.ckpt`
  - vel_6: `outputs/kelvin/xray_vfield/vel_6/nerfstudio_models/step-000004000.ckpt`
  - vel_12: `outputs/kelvin/xray_vfield/vel_12/nerfstudio_models/step-000006000.ckpt`
  - spatiotemporal_mix: `outputs/kelvin/spatiotemporal_mix/vel_12/nerfstudio_models/step-000008000.ckpt`

- **Fixed train script checkpoint naming bug** — `PADSTEPS_VF12_INIT` (NUMSTEPS×2=4000, for refine_vfield output) vs `PADSTEPS_VF12_FINAL` (NUMSTEPS×3=6000, for post-training cp to spatiotemporal_mix). Script now correct for future runs.

- **tiny-cuda-nn-wheels CI matrix expanded** at `https://github.com/igrega348/tiny-cuda-nn-wheels`:
  - Tier 1: cu124 + torch 2.6.0 + py312 for sm_60, 75, 80, 86, 89
  - Tier 2: cu124 + torch 2.6.0 + py311 for sm_75, 80, 86, 89 (Colab default)
  - Tier 1b: cu124 + torch 2.6.0 + py313 for sm_75, 80, 86, 89
  - Tier 3: cu128 + torch 2.8.0 + py312 for sm_86, 89, 90 (Lightning AI)
  - Tier 3b: cu128 + torch 2.8.0 + py313 for sm_86, 89, 90
  - Fixed: `nvidia/cuda:12.8.0-devel-ubuntu22.04` doesn't exist → changed to `12.8.1`
  - 20 jobs total; pipeline triggered and running

- **Evaluation script** `scripts/eval_kelvin.py`:
  - Uses `pipeline.eval_along_plane(plane='xz', distance=0.0, engine='numpy', time=t)` from nerfstudio exporter
  - 4-row figure: vel_6 / vel_12 / spatiotemporal_mix / GT (FEM volume)
  - Separate mixing coefficient α(t) plot
  - Run: `python scripts/eval_kelvin.py --resolution 128`
  - Output: `eval_xsections.png`, `eval_mixing.png`

- **GT intermediate volumes converted** — all 11 timesteps now available as compressed npz (5–7 MB each):
  - `data/kelvin/lattice_00.npz` through `data/kelvin/lattice_20.npz` (even indices, step/20 = normalised time)
  - Converted from `data/kelvin/renders/workshop_local_tXXXX/volume_stage/volume.raw` via `raw_to_npy.py`

- **`.claude/agents/`** — three discoverable agent files: `data-agent.md`, `nerf-agent.md`, `fem-agent.md`

## In Progress / Last Action

Ran `python scripts/eval_kelvin.py --resolution 128` successfully. Figures saved to `eval_xsections.png` and `eval_mixing.png`. Session ended reviewing results.

Key observations from figures:
- vel_6/vel_12: ghosting at intermediate timesteps (two overlapping structures — 50/50 blend of forward+backward canonical at t≠0,1)
- spatiotemporal_mix: clean at t=0,0.2 but abrupt bright transition at t=0.5–1.0
- GT: shows correct progressive compression (top half compresses, bottom fixed)
- α(t) curve: S-shaped, starts ~0.05, ends ~0.99, lags ideal diagonal in t=0.3–0.7 range

## Next Step

Commit all uncommitted changes:
```bash
cd /teamspace/studios/this_studio/NTNU_metamaterials_workshop
git add scripts/train_kelvin_workshop.sh scripts/resize_kelvin_for_eval.sh scripts/eval_kelvin.py
git add .claude/ CLAUDE.md
git add data/kelvin/lattice_0{2,4,6,8}.npz data/kelvin/lattice_1{0,2,4,6,8}.npz
git add neural_xray  # submodule pointer (vfield_config.py, spatiotemporal_mix_config.py)
git commit -m "Add vel_12/spatiotemporal_mix stages, eval script, GT volumes, .claude/agents"
```

Then investigate the spatiotemporal_mix bright-blob artefact at t≥0.5 — may need longer training or different learning rate for the mixing field.

## Failed / Blocked

**Train script checkpoint naming bug (fixed):** The `cp` from vel_12 to spatiotemporal_mix used `step-000004000.ckpt` (the refine_vfield *input*) but vel_12 training saves its final checkpoint at `step-000006000.ckpt`. Fix applied in `scripts/train_kelvin_workshop.sh` — introduced `PADSTEPS_VF12_INIT` / `PADSTEPS_VF12_FINAL`. Spatiotemporal_mix was run manually after the fix.

**canonical_B slow at ~1s/iter** (expected ~65ms): Likely GPU warmup. Stabilised and completed normally.

## Open Questions / Risks

- **spatiotemporal_mix artefact at t≥0.5**: Bright saturated blobs in upper structure at later timesteps. May be insufficient training (only 2000 steps), too-high lr, or the mixing field overcorrecting. Potential fixes: more steps, lower lr, or regularisation on α.
- **Volumetric supervision never fires in vfield stages**: `start_step=3000 > max_iter=2000` — intentional by start_step convention but worth revisiting if reconstruction quality is poor.
- **Neural_xray submodule uncommitted**: `vfield_config.py` and `spatiotemporal_mix_config.py` have config defaults baked in (`distortion_loss_mult=0.0`, `flat_field_loss_multiplier=0.0`, `weight_nn_bias=True`, etc.) — these need committing inside the submodule repo too.
- **tiny-cuda-nn-wheels pipeline**: Check if Tier 3/3b (cu128+torch280) jobs succeed — PyTorch 2.8.0 availability with cu128 index URL needs verification.

## Environment

- GPU: NVIDIA L4 (23 GB VRAM); training uses ~2.9 GB for vfield/mix stages
- Python: `cloudspace` conda env (Python 3.12, PyTorch 2.8+cu128)
- Training: `neural_xray/nerfstudio/nerfstudio/scripts/train.py`
- Editable installs: `neural_xray/nerfstudio`, `neural_xray/nerfstudio-xray/nerf-xray`, `neural_xray/xray_projection_render`
- Branch: `main`
- Training log (from previous run): `/tmp/train_kelvin.log`

## Key Paths

| Artifact | Path | Exists |
|---|---|---|
| Training script | `scripts/train_kelvin_workshop.sh` | ✅ |
| Eval script | `scripts/eval_kelvin.py` | ✅ |
| Resize eval script | `scripts/resize_kelvin_for_eval.sh` | ✅ |
| vfield config | `neural_xray/nerfstudio-xray/nerf-xray/nerf_xray/vfield_config.py` | ✅ |
| spatiotemporal_mix config | `neural_xray/nerfstudio-xray/nerf-xray/nerf_xray/spatiotemporal_mix_config.py` | ✅ |
| canonical_F ckpt | `outputs/kelvin/nerf_xray/canonical_F/nerfstudio_models/step-000002000.ckpt` | ✅ |
| canonical_B ckpt | `outputs/kelvin/nerf_xray/canonical_B/nerfstudio_models/step-000002000.ckpt` | ✅ |
| vel_6 ckpt | `outputs/kelvin/xray_vfield/vel_6/nerfstudio_models/step-000004000.ckpt` | ✅ |
| vel_12 ckpt | `outputs/kelvin/xray_vfield/vel_12/nerfstudio_models/step-000006000.ckpt` | ✅ |
| spatiotemporal_mix ckpt | `outputs/kelvin/spatiotemporal_mix/vel_12/nerfstudio_models/step-000008000.ckpt` | ✅ |
| vel_6 config | `outputs/kelvin/xray_vfield/vel_6/config.yml` | ✅ |
| vel_12 config | `outputs/kelvin/xray_vfield/vel_12/config.yml` | ✅ |
| spatiotemporal_mix config | `outputs/kelvin/spatiotemporal_mix/vel_12/config.yml` | ✅ |
| GT volume t=0 | `data/kelvin/lattice_00.npz` | ✅ |
| GT volume t=0.5 | `data/kelvin/lattice_10.npz` | ✅ |
| GT volume t=1 | `data/kelvin/lattice_20.npz` | ✅ |
| Transforms (canonical F) | `data/kelvin/transforms_00.json` | ✅ |
| Transforms (multi-time) | `data/kelvin/transforms_00_to_20.json` | ✅ |
| Cross-section figure | `eval_xsections.png` | ✅ |
| Mixing coefficient figure | `eval_mixing.png` | ✅ |
| data-agent | `.claude/agents/data-agent.md` | ✅ |
| nerf-agent | `.claude/agents/nerf-agent.md` | ✅ |
| fem-agent | `.claude/agents/fem-agent.md` | ✅ |
