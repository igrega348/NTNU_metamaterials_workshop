# Kelvin-indent X-ray projections

Generate training-ready X-ray projections for Kelvin (or similar) lattice geometries using **`neural_xray/xray_projection_render`** (nested submodule — initialize with `git submodule update --init --recursive`).

## Prerequisites

- [Go](https://go.dev/dl/) 1.21+
- `neural_xray` submodules initialized (includes `xray_projection_render`)

## Inputs

Place analytical scene YAML files here, e.g. `kelvin_indent_t*.yaml` (see [`nerf_data` kelvin example](../neural_xray/nerf_data/scripts/kelvin.yaml)).

## Run

```bash
cd kelvin_indent
./render_files.sh
```

Outputs: `renders/<stem>/images/proj_XX.png`, `transforms.json`, and voxel grids under `renders/<stem>/volume_stage/`.

## Feed into neural_xray

Copy the `renders/` tree into `neural_xray/data/experimental/<your_dataset>/` per the [experimental data workflow](../neural_xray/README.md#experimental-data-workflow), then run `scripts/submit.sh`.
