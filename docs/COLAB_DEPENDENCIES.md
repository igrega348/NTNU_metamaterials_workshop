# Colab dependency stack

The Kelvin Colab notebook uses a **single pinned environment**, not per-package workarounds (`--no-deps`, manual jax installs, etc.).

## Files

| File | Role |
|------|------|
| [`requirements-colab.txt`](../requirements-colab.txt) | Pinned versions for torch, numpy, jax, protobuf |
| [`scripts/install_colab_deps.sh`](../scripts/install_colab_deps.sh) | Installs pins, tinycudann wheel, then editable submodules |
| [`fem_lattice_simulator/pyproject.toml`](../fem_lattice_simulator/pyproject.toml) | `numpy>=1.26,<2.0` (aligned with jax 0.6.2 and nerfstudio) |

## Why numpy 1.26.x (not 2.x)

| Component | numpy constraint |
|-----------|------------------|
| **nerfstudio** (via `nuscenes-devkit`) | `<2.0` |
| **jax 0.6.2** | `>=1.26` (does not require 2.x) |
| **fem_lattice_simulator** | `>=1.26,<2.0` |

Previously FEM declared `numpy>=2.0.0`, which was unnecessary and conflicted with nerfstudio. JAX 0.6.2 only needs numpy ≥1.26.

## Colab runtime

- **Python 3.12** (runtime 2025.10 or 2026.01)
- **T4 GPU** for render/train; **FEM solve uses CPU** (`JAX_PLATFORMS=cpu` in notebook cell 2)
- **Go 1.22.10** at `/usr/local/go` (Colab `apt install golang-go` is 1.18 and cannot build `xray_projection_render`)
- Install cell sets `PYTHON` to the notebook kernel and runs `install_colab_deps.sh`

## Updating pins

1. Change `requirements-colab.txt`.
2. Run the install script on Colab and confirm FEM + a short `train.py` smoke step.
3. If FEM’s `pyproject.toml` changes, commit in `fem_lattice_simulator` and bump the submodule SHA in this repo.

## Eval image downscale (`images_XX_2/`)

Training passes `multi-camera-dataparser --downscale-factors.val 2` (same as `neural_xray/scripts/run_dset.sh`). Val/test splits load **`images_XX_2/eval_00.png`**, not full-res `images_XX/eval_00.png`.

`train_kelvin_workshop.sh` runs `scripts/resize_kelvin_for_eval.sh` before training (same as `neural_xray/scripts/run_dset.sh`).

If you see `FileNotFoundError: .../images_00_2/eval_00.png`, run training again or manually: `bash scripts/resize_kelvin_for_eval.sh` from the repo root.

## Pip “ERROR: dependency conflicts”

Colab’s pre-installed stack may print `ERROR:` during install. Ignore if the script ends with **`All imports OK`**. If you see **`fem-lattice-simulator requires numpy>=2.0`**, restart the runtime and re-run cell 1 (stale pip metadata from an old run).
