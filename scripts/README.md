# scripts/

Helper scripts for setup, data generation, training, and evaluation.

## Setup

| Script | Purpose |
|---|---|
| `init_submodules.sh` | Register and checkout pinned git submodules (`neural_xray`, `fem_lattice_simulator`) after a fresh clone. |
| `install_colab_deps.sh` | Install all Python dependencies in a Colab environment, including the pre-built `tiny-cuda-nn` wheel and editable installs of nerfstudio + nerf-xray. |
| `setup_local.sh` | Create a local conda environment and install the workshop stack (non-Colab). |
| `ensure_go.sh` | Install Go ≥ 1.22 if not present (Colab ships Go 1.18, which is too old for the X-ray renderer). Sourced automatically by `render_projections.sh`. |

## Data pipeline

| Script | Purpose |
|---|---|
| `render_projections.sh` | Render X-ray projections from Kelvin lattice YAML files via the Go renderer. Two-stage per timestep: YAML → voxel grid, then voxel grid → PNG projections. Reads YAMLs from `data/kelvin_indentation/yaml/`, writes renders to `data/kelvin_indentation/renders/`. |
| `stage_kelvin_for_nerf.py` | Stage rendered projections for nerfstudio: renames files to `train_*`/`eval_*`, builds `transforms_*.json` files, and converts `volume.raw` → `lattice_*.npz` for volumetric supervision. Run after `render_projections.sh`. |
| `resize_kelvin_for_eval.sh` | Create downscaled copies of eval images (e.g. `images_00_4/`) for fast validation during training. Called automatically by `train_kelvin_workshop.sh`. |

## Training & evaluation

| Script | Purpose |
|---|---|
| `train_kelvin_workshop.sh` | Run the full 5-stage NeRF training sequence on the Kelvin dataset: canonical forward → canonical backward → velocity field res-6 → velocity field res-12 → spatiotemporal mix. Outputs checkpoints to `outputs/kelvin_indentation/`. |
| `eval_kelvin.py` | Evaluate trained models: renders x-z density cross-sections at selected timesteps for all three model stages plus the GT FEM volume, and plots the spatiotemporal mixing coefficient α(t). Saves `eval_xsections.png` and `eval_mixing.png` to `outputs/kelvin_indentation/`. |
