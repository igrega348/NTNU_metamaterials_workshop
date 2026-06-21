---
name: data-agent
description: Understands the data pipeline for this repo: generating X-ray projections from FEM output, the transforms.json format, time/geometry encoding, the Kelvin lattice rendering pipeline, and staging data for nerfstudio. Use for tasks involving render_projections.sh, stage_kelvin_for_nerf.py, the XRayRenderer API, scene YAMLs, or transforms files.
---

Understands the data pipeline: generating X-ray projections, the transforms file format, and how time/geometry are encoded.

### XRayRenderer API

`xray_projection_render.XRayRenderer` wraps a Go shared library (auto-downloaded from GitHub releases to `~/.cache/xray-renderer/libs/` on first use — no manual build needed).

```python
from xray_projection_render import XRayRenderer

renderer = XRayRenderer()
result = renderer.render({
    'input': 'path/to/scene.yaml',   # YAML describing the 3D object
    'output_dir': 'images/out',
    'resolution': 500,
    'camera_angles': [               # list of dicts with azimuthal/polar (degrees)
        {'azimuthal': 0,  'polar': 90},
        {'azimuthal': 45, 'polar': 90},
    ],
    'R': 4.0,                        # camera distance from origin
    'fov': 40.0,                     # field of view in degrees
})
```

**Speed:** ~3–5 projections/sec on CPU (Go library, no GPU). 16 projections at 500×500 takes ~5 s.

### Scene YAML format

Objects are described as an `object_collection` of primitives:

```yaml
type: object_collection
objects:
  - type: sphere
    center: [-0.5, -0.5, -0.5]
    radius: 0.15
    rho: 1.0            # linear attenuation coefficient
  - type: sphere
    center: [0.5, 0.5, 0.5]
    radius: 0.15
    rho: 1.0
```

`rho` controls X-ray opacity (higher = more attenuating). Coordinates are in world units; the renderer integrates Beer-Lambert attenuation along each ray.

### transforms.json format

Standard nerfstudio-style JSON, extended with a `time` field per frame:

```json
{
  "flat_field": 1,          // flat-field correction value (1 = no correction)
  "camera_angle_x": 0.698,  // horizontal FOV in radians
  "fl_x": 686.87,           // focal length in pixels
  "fl_y": 686.87,
  "w": 500, "h": 500,
  "cx": 250, "cy": 250,     // principal point
  "frames": [
    {
      "file_path": "images_00/train_00.png",
      "time": 0.0,           // normalised time in [0, 1]
      "transform_matrix": [  // 4×4 camera-to-world matrix
        [-1, 0, 6.12e-17, 2.45e-16],
        [6.12e-17, -6.12e-17, 1, 4.0],  // col 3 = camera origin (x,y,z)
        [0, 1, 6.12e-17, 2.45e-16],
        [0, 0, 0, 1]
      ]
    }
  ]
}
```

**transform_matrix layout:**
- `M[:3, :3]` = rotation (camera-to-world), OpenCV convention
- `M[:3, 3]` = camera origin in world coordinates
- Camera looks along its local **+Z** axis; X-ray source is at the origin column

**time field:** normalised float in [0, 1]. `time=0` is the undeformed state, `time=1` is the fully deformed state. All frames at the same timestep share the same `time` value. The multi-timestep file (`transforms_00_to_20.json`) spans t=0.0, 0.1, 0.2, …, 1.0 (11 timesteps, Δt=0.1).

**Note:** the transforms.json field values above (`w/h=500`, `fl_x=686.87`) are from the synthetic balls dataset and are illustrative only. The Kelvin lattice dataset uses `w/h=1024`, `cx/cy=512`, `fl_x≈1406.7`.

### File naming convention

```
data/kelvin/
  images_00/          # timestep 0  (t=0.0)  — 32 train + 1 eval (canonical endpoint)
    train_00.png … train_31.png
    eval_00.png
  images_00_4/        # downscaled eval images (factor 4 → 256×256 from 1024×1024)
    eval_00.png
  images_02/          # timestep 2  (t=0.1)  — 2 train + 1 eval (intermediate)
    train_00.png, train_01.png
    eval_00.png
  images_02_4/        # downscaled eval for intermediate timestep
    eval_00.png
  ...
  images_20/          # last timestep (t=1.0) — 32 train + 1 eval (canonical endpoint)
  transforms_00.json            # single-time file for canonical forward (t=0)
  transforms_20.json            # single-time file for canonical backward (t=1)
  transforms_00_to_20.json      # all timesteps combined
  lattice_00.npz                # voxel grid at t=0 (for volumetric supervision)
  lattice_20.npz                # voxel grid at t=1
```

The `images_XX_4/` folders contain only `eval_*.png` downscaled by factor 4. Created by `scripts/resize_kelvin_for_eval.sh` (called automatically by `train_kelvin_workshop.sh`). The downscale factor is set by `DOWNSCALE_FACTOR=4` in `train_kelvin_workshop.sh`.

Train/eval split is inferred from filename: `train_*` → training set, `eval_*` → eval set.

### Generating data programmatically

See `neural_xray/scripts/generate_data.py`. Pattern:
1. Define sphere positions at each timestep via an analytical deformation field.
2. Dump each timestep's config as a YAML.
3. Call `renderer.render(params)` with that YAML to get PNGs.
4. Accumulate frame dicts (with `time` field) into a master `transforms` JSON.

The two endpoint-only files (`transforms_00.json`, `transforms_20.json`) are used for canonical model training (no `time` needed); the combined file is for the velocity field.

### CLI renderer — for lattice/real data

`scripts/render_projections.sh` has two render paths:

1. **CUDA binary** (default, `USE_CUDA=1`): uses the precompiled `neural_xray/xray_projection_render/xray_render_cuda` binary with `--use_cuda`. Significantly faster than the Go path.
2. **Go CLI** (`USE_CUDA=0`): `go run .` from inside `neural_xray/xray_projection_render/`. Falls back to this if the CUDA binary is absent.

The Go CLI is more configurable for one-off use: it can export voxel grids, accept pre-rasterised volumes, and control projection angles individually.

**Go requirement:** ≥ 1.22. The system `apt` Go (1.18 on Colab) is too old; `scripts/ensure_go.sh` installs 1.22.10 to `/usr/local/go/`. Check with `go version`.

```bash
# From inside neural_xray/xray_projection_render/

# Stage 1: rasterise scene YAML → volume.raw (uint8, flat binary, N³ voxels)
go run . \
  --input scene.yaml \
  --output_dir /tmp/vol_stage/images \
  --num_projections 0 \          # 0 = export only, no projections
  --resolution 128 \             # voxel grid resolution (128³)
  --export_volume \
  --transforms_file /tmp/vol_stage/transforms_volume_export.json \
  --R 4 --fov 40 \
  --text_progress

# Stage 2: render projections from voxel grid
# Write a voxel_grid.yaml descriptor first:
cat > /tmp/vol_stage/voxel_grid.yaml <<EOF
type: voxel_grid
path: /tmp/vol_stage/volume.raw   # absolute path required
resolution: [128, 128, 128]
dtype: uint8
EOF

# Equispaced angles (canonical endpoints — 32 projections):
go run . \
  --input /tmp/vol_stage/voxel_grid.yaml \
  --output_dir /tmp/renders/images \
  --num_projections 32 \
  --resolution 512 \
  --fname_pattern 'proj_%02d.png' \
  --transforms_file /tmp/renders/transforms.json \
  --R 4 --fov 40

# Specific angles (intermediate timesteps — sparse views):
go run . \
  --input /tmp/vol_stage/voxel_grid.yaml \
  --output_dir /tmp/renders/images \
  --azimuthal_angles "0,90,45,135,180,225,270,315" \
  --polar_angles    "90,90,90,90,90,90,90,90" \
  --resolution 512 \
  --fname_pattern 'proj_%02d.png' \
  --transforms_file /tmp/renders/transforms.json \
  --R 4 --fov 40
```

Key CLI flags:

| Flag | Description |
|---|---|
| `--input` | Scene YAML or `voxel_grid.yaml` descriptor |
| `--num_projections N` | N equispaced azimuthal projections at fixed polar; 0 = voxel export only |
| `--azimuthal_angles` / `--polar_angles` | Comma-separated explicit angles (degrees); used instead of `--num_projections` |
| `--resolution` | Image resolution in px (also voxel grid N when `--export_volume`) |
| `--export_volume` | Write `volume.raw` (uint8 flat binary, `N³` bytes) to the parent of `output_dir` (i.e. if `output_dir=/tmp/vol_stage/images`, raw lands at `/tmp/vol_stage/volume.raw`) |
| `--fname_pattern` | Printf pattern for output filenames, e.g. `proj_%02d.png` |
| `--transforms_file` | Path to write the nerfstudio-compatible `transforms.json` |
| `--R` | Camera distance from world origin |
| `--fov` | Horizontal field of view in degrees |

**volume.raw format:** flat uint8, row-major, shape `(N, N, N)`. Size must equal `N³` bytes exactly — `render_projections.sh` validates this. Convert to npz for nerfstudio volumetric supervision via `neural_xray/nerf_data/scripts/raw_to_npy.py`.

### Full Kelvin pipeline (scripts/render_projections.sh)

```bash
# From repo root (defaults shown — VOLUME_RES and RESOLUTION default to 1024):
DATASET_DIR=data/kelvin \
VOLUME_RES=1024 \
RESOLUTION=1024 \
NUM_PROJECTIONS_CANONICAL=32 \
INTERMEDIATE_AZIMUTHAL_ANGLES="0,90,45,135,180,225,270,315" \
USE_CUDA=1 \
bash scripts/render_projections.sh
```

Reads YAMLs from `data/kelvin/yaml/*_t*.yaml`, writes renders to `data/kelvin/renders/<stem>/`.
Two-stage per timestep: YAML→voxel (stage 1), voxel→projections (stage 2). Skips stage 1 if `volume.raw` already exists (use `FORCE_VOXEL_EXPORT=1` to re-export).

### Staging renders for nerfstudio (scripts/stage_kelvin_for_nerf.py)

After rendering, run:
```bash
python scripts/stage_kelvin_for_nerf.py \
  --renders-dir data/kelvin/renders \
  --out-dir data/kelvin \
  --volume-res 1024
```

What it does:
- Renames `proj_XX.png` → `train_XX.png` / `eval_00.png` per the `filename+modulo` split convention
- **Canonical endpoints** (first/last timestep): copy all projections as `train_*`, duplicate `train_00` as `eval_00`
- **Intermediate timesteps**: keep only azimuths 0° and 90° as `train_00`/`train_01`; azimuth 225° as `eval_00`
- Builds `transforms_00.json`, `transforms_T.json` (single-time endpoint files)
- Builds `transforms_00_to_T.json` (combined multi-time file, all timesteps)
- Converts `volume.raw` → `lattice_00.npz` / `lattice_T.npz` via `raw_to_npy.py` (for volumetric supervision)
