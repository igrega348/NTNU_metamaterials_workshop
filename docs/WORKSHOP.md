# NTNU Metamaterials Workshop — 45 min demo

**Format:** A short **live presentation** (~45 minutes) at NTNU (Trondheim, June 2026). Most attendees will **not** run code during the slot; this repo is for **self-paced follow-up** afterward (Colab, local GPU, or lab machine).

## What happens live vs afterward

| Live (45 min) | After the workshop (your own time) |
|---------------|-------------------------------------|
| Motivation: XCT limits, 4D imaging idea | Clone repo + submodules |
| Walk through the **three-stage** pipeline (FEM → X-ray render → neural_xray) | [Colab notebook](../notebooks/00_ntnu_workshop_colab.ipynb) or [local setup](SETUP_LOCAL.md) |
| Show **precomputed** results (TensorBoard, GIFs, slices) if available | Full Kelvin pipeline under `data/kelvin_indentation/` |
| Point to repo URL, Colab badge, paper | Experiment with own lattice / data |

## Suggested 45 min schedule (presenter)

| Time | Topic |
|------|--------|
| 0:00 | Problem: interrupted XCT vs high-speed deformation (metamaterials examples) |
| 0:10 | Method overview: canonical volumes → velocity field → 4D export ([PNAS paper](https://www.pnas.org/doi/10.1073/pnas.2521089122)) |
| 0:20 | **Kelvin workflow:** `fem_lattice_simulator` → `data/kelvin_indentation/` → `neural_xray` |
| 0:30 | Demo **results** (pre-run): projections, training curves, reconstruction |
| 0:38 | How to reproduce: repo layout, Colab badge, pinned submodules |
| 0:45 | Q&A |

## Self-paced pipeline (Kelvin lattice)

```text
fem_lattice_simulator  →  copy YAML to data/kelvin_indentation/yaml/
scripts/render_projections.sh  →  data/kelvin_indentation/renders/
scripts/stage_kelvin_for_nerf.py  →  data/kelvin_indentation/ (transforms, images, lattice_*.npz from volume.raw)
scripts/train_kelvin_workshop.sh  →  outputs/kelvin_indentation/
```

Details: [`data/kelvin_indentation/README.md`](../data/kelvin_indentation/README.md)

Colab: [`notebooks/00_ntnu_workshop_colab.ipynb`](../notebooks/00_ntnu_workshop_colab.ipynb)

## Submodule stack

| Component | Role |
|-----------|------|
| `neural_xray` | Training framework (read-only submodule) |
| `fem_lattice_simulator` | Indentation mechanics |

Pinned commits: [README](../README.md#submodules).

## References

- Grega et al., PNAS 2025 — [High-speed X-ray tomography for 4D imaging](https://www.pnas.org/doi/10.1073/pnas.2521089122)
- Shaikeea et al., Nat. Mater. 2022 — mechanical metamaterials
- Wang et al., PNAS 2024 — interrupted XCT / rubber elasticity

## Troubleshooting (self-paced)

See [SETUP_COLAB.md](SETUP_COLAB.md). `KeyError: 'optimizers'`: delete `outputs/kelvin_indentation/xray_vfield/` and re-run training.
