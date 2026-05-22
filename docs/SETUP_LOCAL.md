# Local installation

Use this if you have an **NVIDIA GPU** and want to run training outside Colab (workstation, NTNU cluster, Lambda, etc.).

## 1. Clone with all submodules

```bash
git clone --recurse-submodules https://github.com/igrega348/NTNU_metamaterials_workshop.git
cd NTNU_metamaterials_workshop
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Top-level submodules: `neural_xray`, `fem_lattice_simulator`. Nested deps under `neural_xray/` are initialized with `--recursive` (see [README](../README.md#submodules)).

If anything is missing:

```bash
bash scripts/init_submodules.sh
```

## 2. Conda environment

From the project root:

```bash
bash scripts/setup_local.sh
```

Or follow the manual steps in [`neural_xray/README.md`](../neural_xray/README.md#local-installation) (Python 3.9, CUDA 12.6, PyTorch 2.6, editable installs of `nerfstudio` and `nerf-xray`).

**Requirements (synthetic demo):** Python ≥3.9, NVIDIA GPU with **≥16 GB** VRAM, CUDA 12.x compatible with PyTorch cu126 wheels.

**Experimental pipeline:** 32 GB+ VRAM recommended; see `neural_xray/scripts/submit.sh`.

## 3. Run the synthetic demo

```bash
conda activate nerfstudio   # name used in upstream docs
cd neural_xray
bash scripts/demo_synthetic.sh
```

Outputs: `neural_xray/outputs/balls/`

TensorBoard:

```bash
tensorboard --logdir neural_xray/outputs/balls/
```

## 4. Experimental data

1. Download datasets from [10.17863/CAM.126862](https://doi.org/10.17863/CAM.126862).
2. Place under `neural_xray/data/experimental/<dataset_name>/` with `transforms_*.json` and volume grids as documented in the upstream README.
3. Run:

```bash
cd neural_xray
bash scripts/submit.sh data/experimental/your_dataset
```

## 5. Updating `neural_xray`

The workshop pins a specific `neural_xray` commit for reproducibility. To move to latest upstream:

```bash
cd neural_xray
git fetch origin
git checkout main
git pull
git submodule update --init --recursive
cd ..
git add neural_xray
git commit -m "Bump neural_xray submodule"
```

Document any commit change in the workshop README for future runs.
