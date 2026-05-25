# Kelvin lattice dataset (`data/kelvin`)

All workshop artifacts for this experiment live under this folder.

```
data/kelvin/
├── yaml/           # FEM-exported scene YAML per timestep (copy or symlink from fem runs)
├── renders/        # X-ray projections from scripts/render_projections.sh
├── transforms_*.json, images_*/, lattice_*.npz   # after scripts/stage_kelvin_for_nerf.py
```

Checkpoints: `outputs/kelvin/` (not here).

## 1. FEM → YAML

```bash
cd fem_lattice_simulator
RUN_NAME=workshop_colab ./run_pipeline.sh   # or run steps manually
cp runs/${RUN_NAME}/yaml/*.yaml ../data/kelvin/yaml/
```

## 2. Render projections

```bash
bash scripts/render_projections.sh
# optional: DATASET_DIR=/path/to/data/kelvin YAML_GLOB='workshop_colab_t*.yaml'
```

| Timestep | Views |
|----------|--------|
| First & last | 32 equispaced (`NUM_PROJECTIONS_CANONICAL`) |
| Intermediate | 8 fixed azimuths (0°, 90°, …, 315°) |

## 3. Stage for neural_xray

```bash
python scripts/stage_kelvin_for_nerf.py \
  --renders-dir data/kelvin/renders \
  --out-dir data/kelvin
```

Writes training layout into this directory (0°/90° train views, 225° eval on intermediates). Canonical grids are `lattice_00.npz` / `lattice_XX.npz` converted from `renders/*/volume_stage/volume.raw` via `neural_xray/nerf_data/scripts/raw_to_npy.py` (same voxel resolution as `VOLUME_RES` in `render_projections.sh`, default 128).

## 4. Train

```bash
bash scripts/train_kelvin_workshop.sh
```

Or [`notebooks/00_ntnu_workshop_colab.ipynb`](../notebooks/00_ntnu_workshop_colab.ipynb).
