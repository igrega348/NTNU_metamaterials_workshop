---
name: fem-agent
description: Understands fem_lattice_simulator/: generating Kelvin lattice FE models, running indentation simulations, and converting deformed outputs to renderer-ready YAMLs. Use for tasks involving run_pipeline.sh, generate_lattice_from_yaml.py, apply_indent_boundary_conditions.py, the FEM solver, or json_to_yaml.py.
---

Understands `fem_lattice_simulator/`: generating Kelvin lattice FE models, running indentation simulations, and converting deformed outputs to renderer-ready YAMLs.

### Overview

JAX-based 3D Euler-Bernoulli beam FEM. Uses `jax.grad` / `jax.hessian` to derive internal forces and tangent stiffness automatically from a strain energy formulation. Solves non-linear indentation via Newton-Raphson. Outputs deformed node coordinates as JSON at each load step, then converts to cylinder-collection YAML for `xray_projection_render`.

**Stack:** JAX (JIT + vmap) for element evaluation, SciPy sparse for assembly and linear solve, meshio for VTK export.

### Installation

```bash
cd fem_lattice_simulator
uv sync           # creates .venv with jax[cpu], scipy, meshio, pyyaml, matplotlib, numpy (no upper bound)
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
  meta/lattice.yaml      # copy of input lattice.yaml for reproducibility
  timesteps/<RUN_NAME>_t0000.json … _tNNNN.json   # deformed node coordinates
  timesteps/<RUN_NAME>_t*.vtu                      # ParaView per-step files
  timesteps/<RUN_NAME>.pvd                         # ParaView collection
  yaml/<RUN_NAME>_t*.yaml   # renderer-ready cylinder YAMLs
```

`run_pipeline.sh` also supports a `YAML_OUTDIR` env var: if set, YAMLs are additionally copied there (with stale files for this run name removed first), avoiding the need to copy manually.

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
  "materials": [{"id": 1, "E": 1.0, "nu": 0.3, "model": "linear_elastic"}],
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

Writes per-step files (with `--output-every 2` and `--ramp-steps 20`: 11 files at t0000, t0002, …, t0020):
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

After `json_to_yaml.py`, copy YAMLs to `data/kelvin_indentation/yaml/`:

```bash
cp runs/my_run/yaml/*.yaml ../../data/kelvin_indentation/yaml/
YAML_GLOB="my_run_t*.yaml" bash ../../scripts/render_projections.sh
```

The renderer identifies first/last timestep from the sorted filenames (alphabetical = chronological for zero-padded `_t####` suffixes) and gives them the full 32-projection canonical treatment; all intermediate steps get the sparse 8-view set.

### Key physics notes

- **Coordinate system:** +z is the indentation axis. Bottom (min z) is fixed; indenter pushes down from max z.
- **Units:** The pipeline uses **non-dimensional units**: `generate_lattice_from_yaml.py` sets `E=1.0`, `nu=0.3`. Coordinates are in normalised units (~[-0.81, 0.81]). Forces and stresses are therefore non-dimensional, not SI Pa/N.
- **Non-linearity:** handled via incremental updated-Lagrangian formulation — reference positions are updated after each converged ramp step (`positions_ref += u_step`), and stiffness is re-assembled at each Newton iteration within a step. Material is linear elastic only.
- **Convergence:** default tolerance `1e-6` (`--tol`), max 20 Newton iterations per step (`--max-iter`). **Increase** `--ramp-steps` if diverging (more steps = smaller increments = better convergence). Additional flags: `--no-vtu` to skip ParaView output, `--no-json` to skip JSON output.
- **Section properties for Kelvin cell:** strut radius 0.025 (YAML units) → A ≈ 1.96e-3, Iy = Iz ≈ 3.07e-7 (non-dimensional, no scaling applied at default `target-half-extent`).
