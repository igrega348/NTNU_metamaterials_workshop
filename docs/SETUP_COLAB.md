# Google Colab setup

Use **after** the 45 min demo. Open [`notebooks/00_ntnu_workshop_colab.ipynb`](../notebooks/00_ntnu_workshop_colab.ipynb), **T4 GPU**, run all cells.

**Pipeline:** FEM → `data/kelvin/yaml/` → `data/kelvin/renders/` → stage into `data/kelvin/` → `outputs/kelvin/`

Install uses pinned versions: [`requirements-colab.txt`](../requirements-colab.txt) and [`scripts/install_colab_deps.sh`](../scripts/install_colab_deps.sh). Details: [`COLAB_DEPENDENCIES.md`](COLAB_DEPENDENCIES.md).

## Troubleshooting

Re-run cell 1 until `install_colab_deps.sh` prints **`All imports OK`** and **`numpy: 1.26.4`**. Python **3.12** runtime required.

**Velocity field restart:**

```bash
rm -rf /content/NTNU_metamaterials_workshop/outputs/kelvin/xray_vfield/
```

Save `outputs/kelvin/` to Drive if the session times out.
