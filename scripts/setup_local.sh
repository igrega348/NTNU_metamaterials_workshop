#!/usr/bin/env bash
# Local environment setup for NTNU_metamaterials_workshop (wraps neural_xray stack).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NX="$ROOT/neural_xray"

if [[ ! -d "$NX/nerfstudio" ]]; then
  echo "Submodules missing. From repo root run:"
  echo "  git submodule update --init --recursive"
  exit 1
fi

ENV_NAME="${NERF_ENV_NAME:-nerfstudio}"

echo "==> Creating conda env: $ENV_NAME (python 3.9)"
conda create --name "$ENV_NAME" -y python=3.9
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install --upgrade pip
conda install -y -c "nvidia/label/cuda-12.6.0" cuda-toolkit
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install ninja
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

echo "==> Installing nerfstudio (editable)"
cd "$NX/nerfstudio"
pip install --upgrade setuptools
pip install -e .

echo "==> Installing nerf-xray (editable)"
cd "$NX/nerfstudio-xray/nerf-xray"
pip install -e .

echo "==> Installing xray_projection_render (editable)"
cd "$NX/xray_projection_render"
pip install -e .

echo ""
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "Run synthetic demo:   cd $NX && bash scripts/demo_synthetic.sh"
