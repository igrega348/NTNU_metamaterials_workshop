# Kelvin lattice dataset (`data/kelvin_indentation`)

All workshop artifacts for this experiment live under this folder.

```
data/kelvin_indentation/
├── yaml/           # FEM-exported scene YAML per timestep (copy or symlink from fem runs)
├── renders/        # X-ray projections from scripts/render_projections.sh
├── transforms_*.json, images_*/, lattice_*.npz   # after scripts/stage_kelvin_for_nerf.py
```

Checkpoints: `outputs/kelvin_indentation/` (not here).

## 1. FEM → YAML

Run from the repo root. `YAML_OUTDIR` copies the exported YAMLs directly into `data/kelvin_indentation/yaml/` — no manual copy needed.

```bash
cd fem_lattice_simulator
YAML_OUTDIR=../data/kelvin_indentation/yaml RUN_NAME=workshop_local ./run_pipeline.sh
cd ..
```

## 2. Render projections

```bash
bash scripts/render_projections.sh
```

Defaults: `VOLUME_RES=1024`, `RESOLUTION=1024`, CUDA renderer if available. Override with env vars.

| Timestep | Views |
|----------|--------|
| First & last | 32 equispaced (`NUM_PROJECTIONS_CANONICAL`) |
| Intermediate | 8 fixed azimuths (0°, 90°, …, 315°) |

## 3. Stage for neural_xray

```bash
python scripts/stage_kelvin_for_nerf.py \
  --renders-dir data/kelvin_indentation/renders \
  --out-dir data/kelvin_indentation
```

Writes training layout into this directory (0°/90° train views, 225° eval on intermediates). Canonical grids are `lattice_00.npz` / `lattice_XX.npz` converted from `renders/*/volume_stage/volume.raw` via `neural_xray/nerf_data/scripts/raw_to_npy.py`. `--volume-res` defaults to 1024 matching `render_projections.sh`.

## 4. Train

```bash
bash scripts/train_kelvin_workshop.sh
```

Or [`notebooks/00_ntnu_workshop_colab.ipynb`](../notebooks/00_ntnu_workshop_colab.ipynb).
