#!/usr/bin/env bash
# Create images_XX_<factor>/ folders with downscaled eval_*.png for val/test splits.
# Matches neural_xray/scripts/run_dset.sh (resize_for_eval.py + multi-camera-dataparser downscale).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NX_ROOT="${WORKSHOP_ROOT}/neural_xray"
RESIZE_SCRIPT="${NX_ROOT}/nerf_data/scripts/resize_for_eval.py"
DATA_DIR="${DATA_DIR:-${WORKSHOP_ROOT}/data/kelvin}"
DOWNSCALE_FACTOR="${DOWNSCALE_FACTOR:-4}"

if [[ ! -f "${RESIZE_SCRIPT}" ]]; then
  echo "error: missing ${RESIZE_SCRIPT}" >&2
  exit 1
fi

shopt -s nullglob
for folder in "${DATA_DIR}"/images_*; do
  [[ -d "${folder}" ]] || continue
  base="$(basename "${folder}")"
  # Skip already-downscaled dirs (e.g. images_00_2).
  if [[ "${base}" =~ ^images_[0-9]+_[0-9]+$ ]]; then
    continue
  fi
  out="${folder}_${DOWNSCALE_FACTOR}"
  if [[ -d "${out}" ]] && compgen -G "${out}/eval_*.png" > /dev/null; then
    echo "  skip ${base} (eval downscale already in $(basename "${out}"))"
    continue
  fi
  if ! compgen -G "${folder}/eval_*.png" > /dev/null; then
    echo "  skip ${base} (no eval_*.png)" >&2
    continue
  fi
  echo "  downscale eval ${DOWNSCALE_FACTOR}x: ${base} -> $(basename "${out}")"
  python "${RESIZE_SCRIPT}" --downscale-factor "${DOWNSCALE_FACTOR}" --folder "${folder}"
done

echo "Eval downscale complete under ${DATA_DIR}/"
