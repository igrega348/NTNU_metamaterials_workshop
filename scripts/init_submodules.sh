#!/usr/bin/env bash
# Register workshop git submodules (pinned SHAs for reproducibility).
# Run from repo root after a plain clone, or if submodules are missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

add_submodule() {
  local path="$1"
  local url="$2"
  local sha="$3"
  if [[ -d "${path}/.git" ]] || [[ -f "${path}/.git" ]]; then
    echo "==> ${path} already present"
    (cd "${path}" && git fetch -q origin && git checkout -q "${sha}")
    return
  fi
  echo "==> Adding ${path} @ ${sha:0:7}"
  git submodule add -f "${url}" "${path}"
  (cd "${path}" && git checkout -q "${sha}")
}

# 4D X-ray stack (includes nested nerfstudio, nerfstudio-xray, nerf_data, xray_projection_render)
add_submodule neural_xray https://github.com/igrega348/neural_xray.git 0181bbb6663df26a35997c72a2f303f8ba00789d

# Lattice FEM (workshop-specific; not bundled inside neural_xray)
add_submodule fem_lattice_simulator https://github.com/igrega348/fem_lattice_simulator.git 471e9b186c1f9aeb61a8bb6eb48f46c7a04e3cc2

echo "==> Initializing nested submodules under neural_xray"
git submodule update --init --recursive neural_xray

echo ""
echo "Done. Submodule status:"
git submodule status
