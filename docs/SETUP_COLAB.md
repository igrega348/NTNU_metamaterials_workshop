# Google Colab setup

Use **after** the 45 min demo. Open [`notebooks/00_ntnu_workshop_colab.ipynb`](../notebooks/00_ntnu_workshop_colab.ipynb), **T4 GPU**, run all cells.

**Pipeline:** FEM → `data/kelvin/yaml/` → `data/kelvin/renders/` → stage into `data/kelvin/` → `outputs/kelvin/`

Nothing is written inside the `neural_xray` submodule.

## Troubleshooting

```bash
rm -rf /content/NTNU_metamaterials_workshop/outputs/kelvin/xray_vfield/
```

Save `outputs/kelvin/` to Drive if the session times out.
