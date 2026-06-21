---
name: nerf-agent
description: Understands the NeRF reconstruction pipeline for this repo: pointing nerfstudio to data, configuring X-ray-specific options, the 5-stage training sequence (canonical_F → canonical_B → vel_6 → vel_12 → spatiotemporal_mix), volumetric supervision, method config defaults, and checkpoint conventions. Use for tasks involving train_kelvin_workshop.sh, vfield_config.py, spatiotemporal_mix_config.py, or any nerfstudio training/eval commands.
---

Understands the reconstruction pipeline: pointing nerfstudio to data, configuring X-ray-specific options, volumetric supervision, and the multi-stage training sequence.

### Methods registered

| Entry point | Class | Use |
|---|---|---|
| `nerf_xray` | `CanonicalPipelineConfig` | Single-time X-ray NeRF (forward or backward canonical volume) |
| `xray_vfield` | `VfieldPipelineConfig` | 4D velocity field trained from both canonical endpoints. Uses `ConstantMixerConfig` (fixed 50/50 blend); `train_deformation_field=True`. |
| `spatiotemporal_mix` | `VfieldPipelineConfig` | Trains a learnable `SpatioTemporalMixerConfig` blending field on top of a **frozen** vel_12 deformation field (`train_deformation_field=False`). |
| `multi-camera-dataparser` | `MultiCameraDataParserConfig` | Dataparser used by all methods |

### Config defaults (baked into method configs)

The following are set in `vfield_config.py` and `spatiotemporal_mix_config.py` and do **not** need to be passed on the CLI:

| Field | Value | File |
|---|---|---|
| `model.distortion_loss_mult` | `0.0` | `vfield_config.py` |
| `model.interlevel_loss_mult` | `0.0` | `vfield_config.py` |
| `pipeline.flat_field_loss_multiplier` | `0.0` | `vfield_config.py`, `spatiotemporal_mix_config.py` |
| `deformation_field.weight_nn_bias` | `True` | `vfield_config.py`, `spatiotemporal_mix_config.py` |
| `deformation_field.weight_nn_gain` | `1.0` | `vfield_config.py`, `spatiotemporal_mix_config.py` |

These were previously `1.0`, `0.002`, `0.001`, `False`, `1e-3` (inherited from NerfactoModelConfig or pipeline defaults).

### Full workshop training pipeline

All 5 stages run sequentially via `scripts/train_kelvin_workshop.sh`. Key variables:

```bash
NUMSTEPS=2000
DOWNSCALE_FACTOR=4      # eval images 256×256 (renders are 1024×1024, factor 4)
BATCH_SIZE=2048         # canonical
BATCH_SIZE_VF=256       # velocity field / mixing
N1=6                    # vel_6 B-spline resolution
N2=12                   # vel_12 B-spline resolution
N_MIX_FIELD=6           # spatiotemporal_mix field_weighing resolution
WEIGHT_NN_WIDTH=20
```

### Stage 1 & 2: Canonical training

```bash
python nerfstudio/nerfstudio/scripts/train.py nerf_xray \
  --data data/kelvin/transforms_00.json \
  --output_dir outputs/ \
  --pipeline.volumetric_supervision True \
  --pipeline.volumetric_supervision_coefficient 1e-3 \
  --pipeline.datamanager.volume_grid_file data/kelvin/lattice_00.npz \
  --pipeline.datamanager.train_num_rays_per_batch 2048 \
  --pipeline.datamanager.eval_num_rays_per_batch 1024 \
  --pipeline.model.eval_num_rays_per_chunk 1024 \
  --pipeline.model.flat_field_trainable False \
  --max-num-iterations 2001 \
  --optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
  --optimizers.fields.scheduler.lr_final 1e-4 \
  --optimizers.fields.scheduler.warmup_steps 50 \
  --optimizers.fields.scheduler.steady_steps 2000 \
  --optimizers.fields.scheduler.max_steps 2000 \
  --timestamp canonical_F \
  multi-camera-dataparser --downscale-factors.val 4 --downscale-factors.test 4
```

Repeat with `transforms_20.json` / `lattice_20.npz` / `--timestamp canonical_B`.

Checkpoints land at:
```
outputs/kelvin/nerf_xray/canonical_F/nerfstudio_models/step-000002000.ckpt
outputs/kelvin/nerf_xray/canonical_B/nerfstudio_models/step-000002000.ckpt
```

### Stage 3: Velocity field res-6

Combine canonical checkpoints, then train:

```bash
python nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py \
  --fwd_ckpt outputs/kelvin/nerf_xray/canonical_F/nerfstudio_models/step-000002000.ckpt \
  --bwd_ckpt outputs/kelvin/nerf_xray/canonical_B/nerfstudio_models/step-000002000.ckpt \
  --out_fn   outputs/kelvin/xray_vfield/vel_6/nerfstudio_models/step-000002000.ckpt

python nerfstudio/nerfstudio/scripts/train.py xray_vfield \
  --data data/kelvin/transforms_00_to_20.json \
  --output_dir outputs/ \
  --load-checkpoint outputs/kelvin/xray_vfield/vel_6/nerfstudio_models/step-000002000.ckpt \
  --load-optimizer False \
  --pipeline.volumetric_supervision True \
  --pipeline.volumetric_supervision_coefficient 1e-4 \
  --pipeline.volumetric_supervision_start_step 3000 \
  --pipeline.datamanager.init_volume_grid_file data/kelvin/lattice_00.npz \
  --pipeline.datamanager.final_volume_grid_file data/kelvin/lattice_20.npz \
  --pipeline.model.deformation_field.num_control_points 6 6 6 \
  --pipeline.model.deformation_field.weight_nn_width 20 \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method matrix \
  --pipeline.model.flat_field_trainable False \
  --pipeline.model.train_field_weighing False \
  --pipeline.model.disable_mixing True \
  --pipeline.datamanager.train_num_rays_per_batch 256 \
  --pipeline.datamanager.eval_num_rays_per_batch 128 \
  --pipeline.model.eval_num_rays_per_chunk 128 \
  --pipeline.model.num_nerf_samples_per_ray 256 \
  --optimizers.fields.optimizer.lr 1e-4 \
  --optimizers.fields.optimizer.weight_decay 1e-1 \
  --optimizers.fields.scheduler.lr_pre_warmup 1e-3 \
  --optimizers.fields.scheduler.lr_final 1e-6 \
  --optimizers.fields.scheduler.warmup_steps 1000 \
  --optimizers.fields.scheduler.steady_steps 1000 \
  --optimizers.fields.scheduler.max_steps 2000 \
  --max-num-iterations 2000 \
  --steps_per_eval_image 500 \
  --steps_per_save 250 \
  --logging.local-writer.max-log-size 10 \
  --timestamp vel_6 \
  --machine.seed 40 \
  multi-camera-dataparser --downscale-factors.val 4 --downscale-factors.test 4
```

Note: `--steps_per_save 250` overrides the method config default of `5000` — without it no intermediate checkpoints are written during vfield training.

### Stage 4: Velocity field res-12

Refine vel_6 → vel_12 checkpoint, then train. **Note: vel_12 checkpoint step = NUMSTEPS×2 = 4000** (naming convention, not actual step count — training still runs for NUMSTEPS=2000 iterations with `--load-optimizer False`).

```bash
python nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py \
  --load-config outputs/kelvin/xray_vfield/vel_6/config.yml \
  --new-resolution 12 \
  --new-nn-width 20 \
  --out-path outputs/kelvin/xray_vfield/vel_12/nerfstudio_models/step-000004000.ckpt

python nerfstudio/nerfstudio/scripts/train.py xray_vfield \
  --data data/kelvin/transforms_00_to_20.json \
  --output_dir outputs/ \
  --load-checkpoint outputs/kelvin/xray_vfield/vel_12/nerfstudio_models/step-000004000.ckpt \
  --load-optimizer False \
  # ... same args as vel_6 but:
  --pipeline.model.deformation_field.num_control_points 12 12 12 \
  --optimizers.fields.scheduler.lr_pre_warmup 1e-3 \
  --optimizers.fields.scheduler.warmup_steps 200 \
  --timestamp vel_12
```

`refine_vfield.py` auto-discovers the latest checkpoint in the vel_6 config's checkpoint dir — no need to specify the checkpoint path explicitly.

### Stage 5: Spatiotemporal mixing

Copies vel_12 checkpoint into a `spatiotemporal_mix/` subdir, then trains the blending field:

```bash
cp outputs/kelvin/xray_vfield/vel_12/nerfstudio_models/step-000004000.ckpt \
   outputs/kelvin/spatiotemporal_mix/vel_12/nerfstudio_models/step-000004000.ckpt

python nerfstudio/nerfstudio/scripts/train.py spatiotemporal_mix \
  --data data/kelvin/transforms_00_to_20.json \
  --output_dir outputs/ \
  --load-checkpoint outputs/kelvin/spatiotemporal_mix/vel_12/nerfstudio_models/step-000004000.ckpt \
  --load-optimizer False \
  --pipeline.volumetric_supervision False \
  --pipeline.model.field_weighing.num_control_points 6 6 6 \
  --pipeline.model.deformation_field.num_control_points 12 12 12 \
  --pipeline.model.deformation_field.weight_nn_width 20 \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method matrix \
  --pipeline.model.flat_field_trainable False \
  --pipeline.model.train_field_weighing True \
  --pipeline.model.disable_mixing False \
  --optimizers.field_weighing.optimizer.lr 1e-2 \
  --optimizers.field_weighing.optimizer.weight_decay 1e-1 \
  --optimizers.field_weighing.scheduler.warmup_steps 200 \
  --optimizers.field_weighing.scheduler.steady_steps 2000 \
  --optimizers.field_weighing.scheduler.max_steps 2000 \
  --timestamp vel_12 \
  --machine.seed 40 \
  multi-camera-dataparser --downscale-factors.val 4 --downscale-factors.test 4
```

Output checkpoint: `outputs/kelvin/spatiotemporal_mix/vel_12/nerfstudio_models/`

### MultiCameraDataParser key options

| Option | Effect |
|---|---|
| `includes_time=True` | Pass the `time` field from JSON to the model (required for `xray_vfield`, `spatiotemporal_mix`) |
| `auto_scale_poses=False` | Do **not** normalise camera positions (X-ray geometry is metric) |
| `center_method='none'` | Do **not** recentre the scene |
| `eval_mode='filename+modulo'` | Split by `train_*` / `eval_*` filename prefix |
| `downscale_factors={'train':1,'val':4}` | Downscale eval images; val images loaded from `images_XX_4/`. Method config default is `val:8`; `4` is the workshop override. |

### Volumetric supervision

When `pipeline.volumetric_supervision=True`, computes a loss between the NeRF's predicted density field and a rasterised voxel grid. Acts as a shape prior. Use `volumetric_supervision_coefficient=1e-3` for canonical, `1e-4` for velocity field.

**Important:** for vel_6 and vel_12, `volumetric_supervision_start_step` is set to `NUMSTEPS+1000=3000`, but training runs for only `NUMSTEPS=2000` iterations. The loss gate is `step > start_step`, so volumetric supervision **never fires** during vfield training — it is intentionally disabled-by-start_step. Disabled explicitly (flag `False`) for `spatiotemporal_mix`.

### Key args for xray_vfield / spatiotemporal_mix

- `--pipeline.model.disable_mixing` — `True` during vfield stages (alternate F/B), `False` for spatiotemporal_mix (blend)
- `--pipeline.model.train_field_weighing` — `False` during vfield (don't train mixer), `True` during spatiotemporal_mix
- `--pipeline.model.deformation_field.num_control_points N N N` — B-spline grid resolution; coarse-to-fine (6→12)
- `--pipeline.model.deformation_field.timedelta` — ODE integration step size for the velocity field
- `--load-optimizer False` — used when loading from combine/refine; resets optimizer state. For vel_6, it's `True` on re-run if the combined ckpt already existed (resume training). For vel_12 and spatiotemporal_mix it's always `False`. Note: `--load-optimizer` only governs optimizer state; step counter comes from the checkpoint's internal step (not the filename).

### Training speed (this environment, Kelvin dataset)

| Stage | Steps | Approx time |
|---|---|---|
| Canonical F or B | 2000 | ~2 min each |
| vel_6 | 2000 | ~6 min |
| vel_12 | 2000 | ~11 min |
| spatiotemporal_mix | 2000 | ~11 min |

### Installation notes (this environment)

- Single conda env (`cloudspace`, Python 3.12, PyTorch 2.8+cu128)
- `tiny-cuda-nn` must be built from source with `--no-build-isolation` (no PyPI wheel for cu128)
- `torch/utils/cpp_extension.py` line ~505: major CUDA version check must be downgraded from `raise` to `logger.warning` because system nvcc=13.0 while PyTorch was built with cu128 — functionally compatible but check fails by default
- `setuptools<71` needed to restore `pkg_resources` (removed in setuptools≥71, required by tiny-cuda-nn setup.py)
- Packages installed editable from submodules: `neural_xray/nerfstudio`, `neural_xray/nerfstudio-xray/nerf-xray`, `neural_xray/xray_projection_render`
