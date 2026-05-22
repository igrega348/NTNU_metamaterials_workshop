# NTNU Metamaterials Workshop — 45 min demo

**Format:** A short **live presentation** (~45 minutes) at NTNU (Trondheim, June 2026). Most attendees will **not** run code during the slot; this repo is for **self-paced follow-up** afterward (Colab, local GPU, or lab machine).

## What happens live vs afterward

| Live (45 min) | After the workshop (your own time) |
|---------------|-------------------------------------|
| Motivation: XCT limits, 4D imaging idea | Clone repo + submodules |
| Walk through the **three-stage** pipeline (FEM → X-ray render → neural_xray) | [Colab notebook](../notebooks/00_ntnu_workshop_colab.ipynb) or [local setup](SETUP_LOCAL.md) |
| Show **precomputed** results (TensorBoard, GIFs, slices) if available | Full **Kelvin indent** pipeline (see below) |
| Point to repo URL, Colab badge, paper | Experiment with own lattice / data |

End-to-end training takes **~45–60+ minutes on a Colab T4** even for the small synthetic demo — too long to run live with a room full of people.

## Suggested 45 min schedule (presenter)

| Time | Topic |
|------|--------|
| 0:00 | Problem: interrupted XCT vs high-speed deformation (metamaterials examples) |
| 0:10 | Method overview: canonical volumes → velocity field → 4D export ([PNAS paper](https://www.pnas.org/doi/10.1073/pnas.2521089122)) |
| 0:20 | **Kelvin workflow (concept):** `fem_lattice_simulator` → `kelvin_indent/render_files.sh` → `neural_xray` |
| 0:30 | Demo **results** (pre-run): projections, training curves, reconstruction — not live install |
| 0:38 | How to reproduce: repo layout, Colab badge, pinned submodules |
| 0:45 | Q&A; link to this repo |

**Presenter tip:** Pre-run at least one path before the talk (Colab or local), save TensorBoard screenshots or short screen recordings, and keep the live segment to storytelling + visuals.

## Learning goals

**During the demo**, attendees should leave able to:

1. Explain why sparse projections + neural rendering can target **dynamic** 3D deformation.
2. Describe the Kelvin pipeline: **mechanics → synthetic XCT → learning**.
3. Know where to find code and instructions to try it later.

**After follow-up**, they should be able to:

4. Run the Colab or local quickstart (synthetic balls or Kelvin data they generated).
5. Locate experimental-data conventions in [`neural_xray`](../neural_xray/README.md) if applying the method to lab data.

## Self-paced pipeline (Kelvin indent)

This is the “interesting synthetic” path discussed in the demo — not required in the 45 min slot:

```text
fem_lattice_simulator/run_pipeline.sh
    → YAML timesteps in runs/<name>/yaml/
kelvin_indent/  (copy YAMLs, run render_files.sh)
    → projections + transforms.json under kelvin_indent/renders/
neural_xray/    (stage under data/experimental/, train via submit.sh or a shortened script)
```

Details: [`kelvin_indent/README.md`](../kelvin_indent/README.md), [`fem_lattice_simulator/README.md`](../fem_lattice_simulator/README.md).

**Rough timings for follow-up (not live):**

| Step | Order of magnitude |
|------|---------------------|
| FEM pipeline | minutes–tens of minutes (mesh size, ramp steps) |
| `render_files.sh` | minutes per timestep (resolution-dependent) |
| neural_xray training | ~30 min (Colab T4, small demo) to hours (full experimental schedule) |

## Before the session (organizer)

- [ ] Repo public; submodules pinned and pushed.
- [ ] Pre-run demo or Kelvin path once; capture figures/video for slides.
- [ ] Slide with: repo URL, [Colab badge](../README.md), paper DOI.
- [ ] Wi-Fi optional for audience (they are not expected to run Colab live).

## After the session (participants)

- [ ] Open [README](../README.md) → Colab or [SETUP_LOCAL.md](SETUP_LOCAL.md).
- [ ] Google account if using Colab; GPU runtime when you have time (~1 h block).
- [ ] For Kelvin: install [Go](https://go.dev/dl/) + `uv`/`pip` for FEM — see submodule READMEs.

## Submodule stack

| Component | Role |
|-----------|------|
| `neural_xray` | Training framework (+ nested `nerfstudio`, `xray_projection_render`, …) |
| `fem_lattice_simulator` | Indentation mechanics on Kelvin lattice |

Pinned submodule commits (see [README](../README.md#submodules)) keep follow-up reproducible even if upstream `main` moves.

## References

- Grega et al., PNAS 2025 — [High-speed X-ray tomography for 4D imaging](https://www.pnas.org/doi/10.1073/pnas.2521089122)
- Shaikeea et al., Nat. Mater. 2022 — mechanical metamaterials
- Wang et al., PNAS 2024 — interrupted XCT / rubber elasticity

## Troubleshooting (self-paced)

See [SETUP_COLAB.md](SETUP_COLAB.md). Common issue when resuming training: `KeyError: 'optimizers'` — delete partial `outputs/.../xray_vfield/` checkpoints and re-run.
