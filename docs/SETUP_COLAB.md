# Google Colab setup (no local install)

Use this **after the NTNU demo** — not during the 45 min session (full run takes ~45–60 min on a T4).

## Option A — Workshop notebook (this repo)

1. Open **[`notebooks/00_ntnu_workshop_colab.ipynb`](../notebooks/00_ntnu_workshop_colab.ipynb)** in Colab:
   - Badge in [README](../README.md), or
   - `https://colab.research.google.com/github/igrega348/NTNU_metamaterials_workshop/blob/main/notebooks/00_ntnu_workshop_colab.ipynb`
2. **Runtime → Change runtime type → T4 GPU** (or better).
3. Run cells top to bottom.

The notebook clones this repository with submodules, installs dependencies, generates synthetic data, and runs the same training pipeline as `neural_xray/scripts/demo_synthetic.sh`.

**Expected runtime (T4):** ~8 min install + ~2 min data + ~30 min training.

## Option B — Upstream demo notebook

The upstream project also ships a full demo (includes extra UI and saved outputs):

- [`neural_xray/colab/demo.ipynb`](../neural_xray/colab/demo.ipynb)
- Colab: `https://colab.research.google.com/github/igrega348/neural_xray/blob/main/colab/demo.ipynb`

That notebook clones `neural_xray` directly from GitHub (not via this workshop repo). Use it if you want the exact PNAS supplementary Colab experience.

## Memory and settings

The synthetic demo is tuned for **~16 GB GPU** (Colab T4):

- Canonical batch size: 2048
- Velocity-field batch size: 256
- Samples per ray: 256

Experimental lattice data typically needs **32 GB+** and the full `submit.sh` schedule; not suitable for free Colab T4.

## Common Colab issues

### `KeyError: 'optimizers'` when restarting velocity-field training

A previous failed run left a partial checkpoint. In the Colab terminal or a new cell:

```bash
rm -rf /content/NTNU_metamaterials_workshop/neural_xray/outputs/balls/xray_vfield/vel_6
rm -rf /content/NTNU_metamaterials_workshop/neural_xray/outputs/balls/xray_vfield/vel_12
```

Then re-run the training cell.

### `tinycudann` / CUDA build errors

The workshop notebook uses **pre-built wheels** for Python 3.12 on Colab (same as upstream demo). Do not mix with a source build of tiny-cuda-nn in the same session.

### Session timeout

Colab may disconnect after ~90 minutes of idle time or when quotas are exceeded. Save important checkpoints to Drive if you extend training beyond the demo.

### Submodule clone fails

If `git clone --recurse-submodules` fails intermittently, re-run the clone cell or use Option B (clone `neural_xray` only).

## After the workshop

- Download `outputs/balls/` from Colab if you want to keep checkpoints.
- For local reproduction, follow [SETUP_LOCAL.md](SETUP_LOCAL.md).
