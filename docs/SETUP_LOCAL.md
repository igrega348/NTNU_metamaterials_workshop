# Local installation

## 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/igrega348/NTNU_metamaterials_workshop.git
cd NTNU_metamaterials_workshop
bash scripts/init_submodules.sh   # if needed
```

## 2. Conda environment

```bash
bash scripts/setup_local.sh
conda activate nerfstudio
```

## 3. Kelvin pipeline

```bash
# FEM → copy YAML to data/kelvin_indentation/yaml/
# then:
bash scripts/render_projections.sh

python scripts/stage_kelvin_for_nerf.py \
  --renders-dir data/kelvin_indentation/renders \
  --out-dir data/kelvin

bash scripts/train_kelvin_workshop.sh
```

- **Data:** `data/kelvin_indentation/`  
- **Outputs:** `outputs/kelvin_indentation/`  
- **Code:** `neural_xray/` submodule only

```bash
tensorboard --logdir outputs/kelvin_indentation/
```

See [`data/kelvin_indentation/README.md`](../data/kelvin_indentation/README.md).
