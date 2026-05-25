#!/usr/bin/env bash
# Install pinned Colab dependencies, then editable workshop/submodule packages.
# Usage (Colab):  PYTHON="$(python -c 'import sys; print(sys.executable)')" bash scripts/install_colab_deps.sh
# Usage (local test): bash scripts/install_colab_deps.sh /path/to/NTNU_metamaterials_workshop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="${1:-${WORKSHOP_ROOT}}"
PYTHON="${PYTHON:-python3}"
PIP="${PYTHON} -m pip"

TCNN_WHEEL="tinycudann-2.0.post75260124-cp312-cp312-linux_x86_64.whl"
TCNN_URL="https://github.com/igrega348/tiny-cuda-nn-wheels/releases/download/1.7.3/${TCNN_WHEEL}"

echo "Workshop root: ${REPO}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import sys; assert sys.version_info[:2] == (3, 12), f'Need Python 3.12, got {sys.version}'"

echo "==> Go (>= 1.22 for xray_projection_render)"
# shellcheck source=ensure_go.sh
source "${SCRIPT_DIR}/ensure_go.sh"
export PATH="${GO_ROOT:-/usr/local/go}/bin:${PATH}"

if ! grep -q 'numpy>=1.26,<2.0' "${REPO}/fem_lattice_simulator/pyproject.toml"; then
  echo "error: fem_lattice_simulator submodule too old — need numpy>=1.26,<2.0" >&2
  exit 1
fi

# Stale editable installs from earlier Colab runs confuse pip (old fem numpy>=2 metadata).
echo "==> Remove stale workshop packages"
${PIP} uninstall -y \
  fem-lattice-simulator nerfstudio nerf-xray xray-renderer tinycudann 2>/dev/null || true

echo "==> Base pins (requirements-colab.txt)"
${PIP} install -q --upgrade pip
${PIP} install -q -r "${REPO}/requirements-colab.txt"

echo "==> tinycudann (prebuilt wheel)"
WHEEL_DIR="${REPO}/.colab-wheels"
mkdir -p "${WHEEL_DIR}"
if [[ "$(uname -s)" == "Linux" ]]; then
  curl -fL "${TCNN_URL}" -o "${WHEEL_DIR}/${TCNN_WHEEL}"
  _bytes="$(wc -c <"${WHEEL_DIR}/${TCNN_WHEEL}" | tr -d ' ')"
  if [[ "${_bytes}" -lt 10000000 ]]; then
    echo "error: tinycudann wheel download failed (${_bytes} bytes)" >&2
    exit 1
  fi
  ${PIP} install -q "${WHEEL_DIR}/${TCNN_WHEEL}" --force-reinstall
else
  echo "warning: skipping tinycudann wheel on non-Linux host ($(uname -s))" >&2
fi

echo "==> Editable: nerfstudio"
cd "${REPO}/neural_xray/nerfstudio"
${PIP} install -q -e .

echo "==> Editable: nerf-xray"
cd "${REPO}/neural_xray/nerfstudio-xray/nerf-xray"
${PIP} install -q -e .

echo "==> Editable: xray-renderer"
cd "${REPO}/neural_xray/xray_projection_render"
${PIP} install -q -e .

echo "==> Editable: fem-lattice-simulator"
cd "${REPO}/fem_lattice_simulator"
${PIP} install -q -e .

echo "==> Verify versions"
"${PYTHON}" - <<'PY'
import importlib.metadata as md
import numpy as np

pkgs = ("nerfstudio", "nerf-xray", "fem-lattice-simulator", "xray-renderer", "jax", "jaxlib")
for name in pkgs:
    print(f"  {name}: {md.version(name)}")
print(f"  numpy: {np.__version__}")
if not np.__version__.startswith("1.26"):
    raise SystemExit(f"expected numpy 1.26.x, got {np.__version__}")

import tinycudann  # noqa: F401
import nerfstudio
import nerf_xray
import jax  # noqa: F401

print("All imports OK")
PY

echo "Done."
