# Agent Context: NTNU Metamaterials Workshop

Three specialised agents for working with this repo.

---

## data-agent

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

**time field:** normalised float in [0, 1]. `time=0` is the undeformed state, `time=1` is the fully deformed state. All frames at the same timestep share the same `time` value. The multi-timestep file (`transforms_00_to_20.json`) spans t=0.0, 0.05, 0.1, …, 1.0 (21 steps × varying projections per step).

### File naming convention

```
data/synthetic/balls/
  images_00/          # timestep 0  (t=0.0)  — 16 train + 1 eval
    train_00.png … train_15.png
    eval_00.png
  images_01/          # timestep 1  (t=0.05) — 2 train + 1 eval
    train_00.png, train_01.png
    eval_00.png
  ...
  images_20/          # timestep 20 (t=1.0)  — 16 train + 1 eval
  transforms_00.json            # single-time file for canonical forward (t=0)
  transforms_20.json            # single-time file for canonical backward (t=1)
  transforms_00_to_20.json      # all timesteps combined (93 frames total)
  balls_00.yaml                 # volume YAML at t=0 (for volumetric supervision)
  balls_20.yaml                 # volume YAML at t=1
```

Train/eval split is inferred from filename: `train_*` → training set, `eval_*` → eval set.

### Generating data programmatically

See `neural_xray/scripts/generate_data.py`. Pattern:
1. Define sphere positions at each timestep via an analytical deformation field.
2. Dump each timestep's config as a YAML.
3. Call `renderer.render(params)` with that YAML to get PNGs.
4. Accumulate frame dicts (with `time` field) into a master `transforms` JSON.

The two endpoint-only files (`transforms_00.json`, `transforms_20.json`) are used for canonical model training (no `time` needed); the combined file is for the velocity field.

### CLI renderer (`go run .`) — for lattice/real data

The Go CLI is used directly by `scripts/render_projections.sh` for the Kelvin lattice pipeline. It is more powerful than the Python API: it can export voxel grids, accept pre-rasterised volumes, and control projection angles individually.

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
| `--export_volume` | Write `volume.raw` (uint8 flat binary, `N³` bytes) to `output_dir/../volume_stage/` |
| `--fname_pattern` | Printf pattern for output filenames, e.g. `proj_%02d.png` |
| `--transforms_file` | Path to write the nerfstudio-compatible `transforms.json` |
| `--R` | Camera distance from world origin |
| `--fov` | Horizontal field of view in degrees |

**volume.raw format:** flat uint8, row-major, shape `(N, N, N)`. Size must equal `N³` bytes exactly — `render_projections.sh` validates this. Convert to npz for nerfstudio volumetric supervision via `neural_xray/nerf_data/scripts/raw_to_npy.py`.

### Full Kelvin pipeline (scripts/render_projections.sh)

```bash
# From repo root:
DATASET_DIR=data/kelvin \
VOLUME_RES=128 \
RESOLUTION=512 \
NUM_PROJECTIONS_CANONICAL=32 \
INTERMEDIATE_AZIMUTHAL_ANGLES="0,90,45,135,180,225,270,315" \
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
  --volume-res 128
```

What it does:
- Renames `proj_XX.png` → `train_XX.png` / `eval_00.png` per the `filename+modulo` split convention
- **Canonical endpoints** (first/last timestep): copy all projections as `train_*`, duplicate `train_00` as `eval_00`
- **Intermediate timesteps**: keep only azimuths 0° and 90° as `train_00`/`train_01`; azimuth 225° as `eval_00`
- Builds `transforms_00.json`, `transforms_T.json` (single-time endpoint files)
- Builds `transforms_00_to_T.json` (combined multi-time file, all timesteps)
- Converts `volume.raw` → `lattice_00.npz` / `lattice_T.npz` via `raw_to_npy.py` (for volumetric supervision)

---

## nerf-agent

Understands the reconstruction pipeline: pointing nerfstudio to data, configuring X-ray-specific options, volumetric supervision, and the multi-stage training sequence.

### Methods registered

| Entry point | Class | Use |
|---|---|---|
| `nerf_xray` | `CanonicalPipelineConfig` | Single-time X-ray NeRF (forward or backward canonical volume) |
| `xray_vfield` | `VfieldPipelineConfig` | 4D velocity field trained from both canonical endpoints |
| `multi-camera-dataparser` | `MultiCameraDataParserConfig` | Dataparser used by both methods |

### Canonical training (Steps 2 & 3)

```bash
python nerfstudio/nerfstudio/scripts/train.py nerf_xray \
  --data path/to/transforms_00.json \   # single-time transforms file
  --output_dir outputs/ \
  --pipeline.datamanager.volume_grid_file path/to/balls_00.yaml \
  --pipeline.volumetric_supervision False \
  --pipeline.datamanager.train_num_rays_per_batch 2048 \
  --pipeline.model.flat_field_trainable False \
  --max-num-iterations 2001 \
  --timestamp canonical_F \
  multi-camera-dataparser --downscale-factors.val 2 --downscale-factors.test 2
```

Key args:
- `--data` — path to a **single-timestep** transforms JSON
- `--pipeline.datamanager.volume_grid_file` — YAML describing the 3D volume (used for volumetric supervision loss if enabled)
- `--pipeline.volumetric_supervision` — `True` adds a loss term matching the NeRF density field to the known YAML geometry; set `False` when no ground-truth volume is available
- `--pipeline.model.flat_field_trainable` — whether to learn a per-ray flat-field correction (detector sensitivity map); `False` for synthetic data with `flat_field=1`
- `--timestamp` — subfolder name for the checkpoint, e.g. `canonical_F` / `canonical_B`

Checkpoints land at:
```
outputs/<dataset_name>/nerf_xray/canonical_F/nerfstudio_models/step-XXXXXXXXX.ckpt
```

### Velocity field training (Steps 4 & 5)

First combine forward + backward checkpoints:
```bash
python nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py \
  --fwd_ckpt outputs/.../canonical_F/nerfstudio_models/step-000002000.ckpt \
  --bwd_ckpt outputs/.../canonical_B/nerfstudio_models/step-000002000.ckpt \
  --out_fn   outputs/.../xray_vfield/vel_6/nerfstudio_models/step-000002000.ckpt
```

Then train:
```bash
python nerfstudio/nerfstudio/scripts/train.py xray_vfield \
  --data path/to/transforms_00_to_20.json \   # multi-time combined file
  --output_dir outputs/ \
  --load-checkpoint outputs/.../vel_6/nerfstudio_models/step-000002000.ckpt \
  --pipeline.datamanager.init_volume_grid_file  balls_00.yaml \
  --pipeline.datamanager.final_volume_grid_file balls_20.yaml \
  --pipeline.model.deformation_field.num_control_points 6 6 6 \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method matrix \
  --max-num-iterations 2000 \
  --timestamp vel_6 \
  multi-camera-dataparser --downscale-factors.val 2 --downscale-factors.test 2
```

Then refine to res-12:
```bash
python nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py \
  --load-config outputs/.../xray_vfield/vel_6/config.yml \
  --new-resolution 12 \
  --new-nn-width 20 \
  --out-path outputs/.../xray_vfield/vel_12/nerfstudio_models/step-000004000.ckpt
```

Key args for `xray_vfield`:
- `--data` — the **multi-timestep** transforms file (`transforms_00_to_20.json`); `time` field in each frame is passed to the deformation field
- `--pipeline.datamanager.init_volume_grid_file` / `final_volume_grid_file` — YAMLs at t=0 and t=1 for volumetric supervision
- `--pipeline.model.deformation_field.num_control_points N N N` — B-spline grid resolution; coarse-to-fine (6→12)
- `--pipeline.model.deformation_field.timedelta` — integration step size for the velocity field ODE solver
- `--pipeline.volumetric_supervision_coefficient` — weight of the volumetric loss (e.g. `1e-4`)
- `--pipeline.volumetric_supervision_start_step` — delay volumetric supervision until this step (let the network warm up first)

### MultiCameraDataParser key options

| Option | Effect |
|---|---|
| `includes_time=True` | Pass the `time` field from JSON to the model (required for `xray_vfield`) |
| `auto_scale_poses=False` | Do **not** normalise camera positions (X-ray geometry is metric) |
| `center_method='none'` | Do **not** recentre the scene |
| `eval_mode='filename+modulo'` | Split by `train_*` / `eval_*` filename prefix |
| `downscale_factors={'train':1,'val':2}` | Downscale eval images to speed up validation |

### Volumetric supervision

When `pipeline.volumetric_supervision=True`, the pipeline computes a loss between the NeRF's predicted density field and a rasterised version of the YAML volume. The YAML is rasterised to a voxel grid and compared to sigma values sampled from the network. This acts as a shape prior, preventing the network from placing density in unoccupied regions. Use with `volumetric_supervision_coefficient=1e-4` — too high causes the network to ignore the X-ray data.

### Training speed (L4 GPU, synthetic balls dataset)

| Stage | Steps | ms/iter | Total |
|---|---|---|---|
| Canonical (fwd/bwd) | 2000 | ~65 ms | ~2 min each |
| Velocity field res-6 | 2000 | ~175 ms | ~6 min |
| Velocity field res-12 | 2000 | ~335 ms | ~11 min |

Final losses: canonical ~0.020, vfield res-6 ~0.13, vfield res-12 ~0.20 (higher because res-12 is regularising a more expressive field).

### Installation notes (this environment)

- Single conda env (`cloudspace`, Python 3.12, PyTorch 2.8+cu128)
- `tiny-cuda-nn` must be built from source with `--no-build-isolation` (no PyPI wheel for cu128)
- `torch/utils/cpp_extension.py` line ~505: major CUDA version check must be downgraded from `raise` to `logger.warning` because system nvcc=13.0 while PyTorch was built with cu128 — functionally compatible but check fails by default
- `setuptools<71` needed to restore `pkg_resources` (removed in setuptools≥71, required by tiny-cuda-nn setup.py)
- Packages installed editable from submodules: `neural_xray/nerfstudio`, `neural_xray/nerfstudio-xray/nerf-xray`, `neural_xray/xray_projection_render`

---

## fem-agent

Understands `fem_lattice_simulator/`: generating Kelvin lattice FE models, running indentation simulations, and converting deformed outputs to renderer-ready YAMLs.

### Overview

JAX-based 3D Euler-Bernoulli beam FEM. Uses `jax.grad` / `jax.hessian` to derive internal forces and tangent stiffness automatically from a strain energy formulation. Solves non-linear indentation via Newton-Raphson. Outputs deformed node coordinates as JSON at each load step, then converts to cylinder-collection YAML for `xray_projection_render`.

**Stack:** JAX (JIT + vmap) for element evaluation, SciPy sparse for assembly and linear solve, meshio for VTK export.

### Installation

```bash
cd fem_lattice_simulator
uv sync           # creates .venv with jax, scipy, meshio, pyyaml, numpy<2
source .venv/bin/activate
# or: pip install -e .
```

### Full pipeline (run_pipeline.sh)

```bash
cd fem_lattice_simulator
RUN_NAME=my_run ./run_pipeline.sh
```

Four steps:
1. **Generate lattice JSON** from `lattice.yaml` (tessellate unit cell → FEM model)
2. **Apply boundary conditions** (bottom roller + top indenter patch)
3. **Solve** (Newton-Raphson ramp, write timestep JSONs)
4. **Convert** deformed JSONs → cylinder YAMLs for renderer

Outputs land in `runs/<RUN_NAME>/`:
```
runs/<RUN_NAME>/
  model/out.json         # FEM model with BCs
  timesteps/<RUN_NAME>_t0000.json … _tNNNN.json   # deformed node coordinates
  yaml/<RUN_NAME>_t*.yaml   # renderer-ready cylinder YAMLs
```

### lattice.yaml — unit cell definition

```yaml
type: tessellated_obj_coll
uc:                          # unit cell
  type: unit_cell
  xmin: 0.0  xmax: 0.4      # bounding box of one unit cell
  ymin: 0.0  ymax: 0.4
  zmin: 0.0  zmax: 0.4
  objects:
    type: object_collection
    objects:
      - type: cylinder
        p0: [0.1, 0.0, 0.2]   # start point
        p1: [0.2, 0.0, 0.3]   # end point
        radius: 0.025          # strut radius (used for section props)
        rho: 1                 # X-ray attenuation (passed through to renderer YAML)
      # ... more cylinders (Kelvin cell has ~36 struts per UC)
xmin: -0.81  xmax: 0.81      # overall bounding box of tessellated structure
ymin: -0.81  ymax: 0.81
zmin: -0.81  zmax: 0.81
```

The tessellation is controlled by `--nx --ny --nz` in `generate_lattice_from_yaml.py`.

### Step 1: generate_lattice_from_yaml.py

```bash
uv run python scripts/generate_lattice_from_yaml.py \
  --yaml lattice.yaml \
  --out runs/my_run/model/out.json \
  --nx 4 --ny 4 --nz 4 \    # number of unit cells per axis
  --subdivide 8              # beam subdivision (elements per strut; higher = more accurate)
```

- Reads `lattice.yaml`, tiles the unit cell `nx×ny×nz` times
- Scales coordinates to `~[-1,1]³` (half-extent 0.8 by default via `--target-half-extent`)
- Derives circular beam section properties from cylinder `radius` (A, Iy, Iz, J)
- Stores `meta.unit_cell_period` in the JSON (needed by `apply_indent_boundary_conditions.py`)
- Colab uses `--nx 4 --ny 4 --nz 4 --subdivide 4` (faster); desktop uses `--subdivide 8`

### lattice.json — FEM model format

```json
{
  "materials": [{"id": 1, "E": 210e9, "nu": 0.3, "model": "linear_elastic"}],
  "sections":  [{"id": 1, "A": 0.00196, "Iy": 3.07e-7, "Iz": 3.07e-7, "J": 6.14e-7}],
  "nodes":     [{"id": 1, "coords": [x, y, z]}, ...],
  "elements":  [{"id": 1, "nodes": [1, 2], "material": 1, "section": 1}, ...],
  "boundary_conditions": [
    {"node": 5, "dof": ["ux","uy","uz","rx","ry","rz"], "value": 0.0}
  ],
  "point_loads": [
    {"node": 3, "dof": "fz", "value": -1000.0}
  ]
}
```

DOFs per node: `ux uy uz rx ry rz` (3 translations + 3 rotations).

### Step 2: apply_indent_boundary_conditions.py

```bash
uv run python scripts/apply_indent_boundary_conditions.py \
  --in  runs/my_run/model/out.json \
  --out runs/my_run/model/out.json \
  --patch-cells-x 2 --patch-cells-y 2 \   # indenter footprint in unit cells
  --patch-placement center \               # or "origin"
  --indent-uz -0.8 \                       # prescribed displacement (negative = downward)
  --indenter-uxuy-zero                     # also fix lateral DOFs under indenter
```

- Bottom plane (min z): roller BCs (`uz=0`)
- Top plane (max z), patch region: prescribed `uz` (ramped 0 → `indent-uz` during solve)
- `--cell-size` defaults to `meta.unit_cell_period[0]` from the JSON if present

### Step 3: solver (src/main.py)

```bash
uv run python -m src.main runs/my_run/model/out.json \
  --ramp-steps 20 \           # number of load increments (t=0…20)
  --output-prefix runs/my_run/timesteps/my_run \
  --output-every 2            # export JSON+VTU every 2 steps
```

Writes per-step files:
- `my_run_t0000.json` … `my_run_t0020.json` — deformed node coordinates + strains/stresses
- `my_run_t0000.vtu` … — ParaView files (apply "Warp By Vector" to visualise deformation)
- `my_run.pvd` — ParaView collection file

`--output-steps 0,10,20` exports only specific steps. `--timestep-mode step` writes integer step index as VTK time (default `factor` writes t/T_max normalised).

### Step 4: json_to_yaml.py

```bash
uv run python scripts/json_to_yaml.py \
  "runs/my_run/timesteps/my_run_t*.json" \
  --radius-from-area \    # infer cylinder radius from section A (r = sqrt(A/π))
  --outdir runs/my_run/yaml \
  --overwrite
```

- Reads deformed node coords from each timestep JSON
- Writes one `object_collection` YAML per timestep (cylinders at deformed positions)
- `--radius-from-area`: derive display radius from section area (matches original `lattice.yaml` geometry)
- Output naming: `my_run_t0000.yaml`, `my_run_t0002.yaml`, … (matching the JSON input stems)
- These YAMLs feed directly into `scripts/render_projections.sh` via `YAML_GLOB=my_run_t*.yaml`

### Connecting FEM → renderer

After `json_to_yaml.py`, copy YAMLs to `data/kelvin/yaml/`:

```bash
cp runs/my_run/yaml/*.yaml ../../data/kelvin/yaml/
YAML_GLOB="my_run_t*.yaml" bash ../../scripts/render_projections.sh
```

The renderer identifies first/last timestep from the sorted filenames (alphabetical = chronological for zero-padded `_t####` suffixes) and gives them the full 32-projection canonical treatment; all intermediate steps get the sparse 8-view set.

### Key physics notes

- **Coordinate system:** +z is the indentation axis. Bottom (min z) is fixed; indenter pushes down from max z.
- **Units:** SI (Pa, m) in `lattice.json`. `lattice.yaml` uses normalised coordinates (~[-0.81, 0.81]); `generate_lattice_from_yaml.py` handles the scale mapping.
- **Non-linearity:** geometric non-linearity is handled by re-assembling stiffness at each Newton step. Material is linear elastic only.
- **Convergence:** default tolerance `1e-6`, max 20 Newton iterations per step. Reduce `--ramp-steps` if diverging (finer increments).
- **Section properties for Kelvin cell:** default strut radius 0.025 (in YAML units) → A ≈ 1.96e-3 m², Iy = Iz ≈ 3.07e-7 m⁴ after normalisation scaling.
