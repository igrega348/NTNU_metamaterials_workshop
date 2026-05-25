# NTNU Metamaterials Workshop — Neural X-ray 4D Tomography

Materials for a **~45 min live demo** at **NTNU (Trondheim, June 2026)** on neural rendering for **4D X-ray tomography** of architected materials — plus everything to **reproduce the pipeline afterward** (Colab or local GPU). The session itself is presentation-focused; expect **~1 hour+** if you run training on your own time.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/igrega348/NTNU_metamaterials_workshop/blob/main/notebooks/00_ntnu_workshop_colab.ipynb)
[![Paper](https://img.shields.io/badge/Paper-PNAS-blue)](https://www.pnas.org/doi/10.1073/pnas.2521089122)
[![Upstream](https://img.shields.io/badge/Code-neural__xray-green)](https://github.com/igrega348/neural_xray)

## What is included

| Path | Description |
|------|-------------|
| [`neural_xray/`](neural_xray/) | 4D X-ray neural rendering (**submodule**, includes `xray_projection_render` nested) |
| [`fem_lattice_simulator/`](fem_lattice_simulator/) | Lattice FEM (**submodule**) |
| [`data/kelvin/`](data/kelvin/) | FEM YAML, renders, staged training data |
| [`notebooks/`](notebooks/) | Lightweight **Colab** notebook (no local install) |
| [`docs/`](docs/) | Workshop agenda, Colab/local setup |
| [`data/`](data/) | Training datasets (generated; not inside submodules) |
| [`outputs/`](outputs/) | Checkpoints and logs from `train_kelvin_workshop.sh` |
| [`scripts/`](scripts/) | `render_projections.sh`, staging, training, `setup_local.sh` |

Submodules are pinned for **code only**. Generated data and models go under [`data/`](data/) and [`outputs/`](outputs/) in this repo — never inside `neural_xray/`.

## Try it after the demo (Colab)

1. Open the [Kelvin pipeline notebook](notebooks/00_ntnu_workshop_colab.ipynb) (badge above).
2. **Runtime → T4 GPU** (or better).
3. Run all cells: FEM → X-ray render → stage transforms → train.

Expect **1–3+ hours** end-to-end on Colab. Details: [`docs/SETUP_COLAB.md`](docs/SETUP_COLAB.md)

## Quick start (local GPU)

```bash
git clone --recurse-submodules https://github.com/igrega348/NTNU_metamaterials_workshop.git
cd NTNU_metamaterials_workshop
bash scripts/setup_local.sh
conda activate nerfstudio
# … FEM + render + stage (see data/kelvin/README.md), then:
bash scripts/train_kelvin_workshop.sh
```

Details: [`docs/SETUP_LOCAL.md`](docs/SETUP_LOCAL.md)

## Live demo outline

See [`docs/WORKSHOP.md`](docs/WORKSHOP.md) for the **45 min presenter schedule**, what to show pre-run, and self-paced follow-up steps.

## Submodules

Clone with nested dependencies:

```bash
git clone --recurse-submodules https://github.com/igrega348/NTNU_metamaterials_workshop.git
```

If needed: `bash scripts/init_submodules.sh`

| Submodule | Role |
|-----------|------|
| [`neural_xray`](https://github.com/igrega348/neural_xray) | Training stack (also pulls in `nerfstudio`, `nerfstudio-xray`, `nerf_data`, `xray_projection_render`) |
| [`fem_lattice_simulator`](https://github.com/igrega348/fem_lattice_simulator) | Kelvin lattice indentation FEM |

Each submodule is pinned to a commit in this repo for reproducible follow-up.

## Upstream project

This workshop builds on **[neural_xray](https://github.com/igrega348/neural_xray)** — neural rendering for 4D X-CT (PNAS 2025). The submodule pin ensures all participants use the same code revision. To cite the method:

```bibtex
@article{grega2024highspeed,
  title={High-speed X-ray tomography for 4D imaging},
  author={Ivan Grega and William Whitney and Vikram Sudhir Deshpande},
  journal={Proceedings of the National Academy of Sciences},
  year={2025},
  doi={10.1073/pnas.2521089122}
}
```

## License

MIT — see [LICENSE](LICENSE). The `neural_xray` submodule retains its own license (MIT).
