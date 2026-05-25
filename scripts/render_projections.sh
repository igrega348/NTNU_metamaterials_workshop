#!/usr/bin/env bash
# Two-stage lattice X-ray pipeline using neural_xray/xray_projection_render.
#
# Reads scene YAML from data/<dataset>/yaml/, writes renders to data/<dataset>/renders/.
#
#   1) YAML -> UINT8 voxel grid (volume.raw)
#   2) voxel grid -> projections (transforms.json + proj_XX.png)
#
# Projection policy:
#   First & last timestep: NUM_PROJECTIONS_CANONICAL equispaced (default 32)
#   Intermediate: INTERMEDIATE_AZIMUTHAL_ANGLES (default 8 views; train 0°/90°, eval 225° after staging)
#
# Environment:
#   DATASET_DIR      default: <repo>/data/kelvin
#   YAML_GLOB        under data/<dataset>/yaml/ (default: *_t*.yaml)
#   VOLUME_RES, RESOLUTION, NUM_PROJECTIONS_CANONICAL, INTERMEDIATE_* , CAMERA_R, FOV_DEG
#   FORCE_VOXEL_EXPORT=1
#
# Options: --clear  removes data/<dataset>/renders/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Colab: apt golang-go is 1.18; prefer /usr/local/go from ensure_go.sh / install_colab_deps.sh
if [[ -x /usr/local/go/bin/go ]]; then
  export PATH="/usr/local/go/bin:${PATH}"
fi
WORKSHOP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-${WORKSHOP_ROOT}/data/kelvin}"
YAML_DIR="${DATASET_DIR}/yaml"
RENDERS_DIR="${DATASET_DIR}/renders"
XRAY_DIR="${WORKSHOP_ROOT}/neural_xray/xray_projection_render"

usage() {
  echo "usage: ${0##*/} [--clear]" >&2
  echo "  Dataset directory: ${DATASET_DIR}" >&2
  echo "  YAML inputs:       ${YAML_DIR}/<YAML_GLOB>" >&2
  echo "  Renders output:    ${RENDERS_DIR}/" >&2
}

do_clear=0
for arg in "$@"; do
  case "${arg}" in
    --clear) do_clear=1 ;;
    -h | --help) usage; exit 0 ;;
    *)
      echo "error: unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${do_clear}" -eq 1 ]]; then
  if [[ -d "${RENDERS_DIR}" ]]; then
    rm -rf "${RENDERS_DIR}"
    echo "Removed ${RENDERS_DIR}"
  else
    echo "Nothing to remove (${RENDERS_DIR} is missing)"
  fi
  exit 0
fi

VOLUME_RES="${VOLUME_RES:-128}"
RESOLUTION="${RESOLUTION:-512}"
NUM_PROJECTIONS_CANONICAL="${NUM_PROJECTIONS_CANONICAL:-32}"
INTERMEDIATE_AZIMUTHAL_ANGLES="${INTERMEDIATE_AZIMUTHAL_ANGLES:-0,90,45,135,180,225,270,315}"
INTERMEDIATE_POLAR_ANGLES="${INTERMEDIATE_POLAR_ANGLES:-90,90,90,90,90,90,90,90}"
CAMERA_R="${CAMERA_R:-4}"
FOV_DEG="${FOV_DEG:-40}"
FORCE_VOXEL_EXPORT="${FORCE_VOXEL_EXPORT:-0}"
YAML_GLOB="${YAML_GLOB:-*_t*.yaml}"

if [[ ! -d "${XRAY_DIR}" ]]; then
  echo "error: xray_projection_render not found at ${XRAY_DIR}" >&2
  exit 1
fi

mkdir -p "${YAML_DIR}" "${RENDERS_DIR}"

shopt -s nullglob
yaml_files=("${YAML_DIR}"/${YAML_GLOB})
shopt -u nullglob

if [[ ${#yaml_files[@]} -eq 0 ]]; then
  echo "error: no files matching ${YAML_GLOB} under ${YAML_DIR}" >&2
  exit 1
fi

IFS=$'\n' yaml_files=($(sort <<<"${yaml_files[*]}"))
unset IFS

FIRST_STEM="$(basename "${yaml_files[0]}" .yaml)"
LAST_STEM="$(basename "${yaml_files[${#yaml_files[@]} - 1]}" .yaml)"

echo "Dataset: ${DATASET_DIR}"
echo "Pipeline (${YAML_GLOB}): YAML -> volume.raw (${VOLUME_RES}^3) @ ${RESOLUTION}px"
echo "  canonical (${FIRST_STEM}, ${LAST_STEM}): ${NUM_PROJECTIONS_CANONICAL} projections"
echo "  intermediate: azimuth ${INTERMEDIATE_AZIMUTHAL_ANGLES}"
echo "Renderer: ${XRAY_DIR}"

cd "${XRAY_DIR}"

for yaml_path in "${yaml_files[@]}"; do
  stem="$(basename "${yaml_path}" .yaml)"
  out_root="${RENDERS_DIR}/${stem}"
  vol_stage="${out_root}/volume_stage"
  vol_images="${vol_stage}/images"
  final_images="${out_root}/images"
  volume_raw="${vol_stage}/volume.raw"
  voxel_desc="${vol_stage}/voxel_grid.yaml"

  mkdir -p "${vol_images}" "${final_images}"

  echo "==> ${stem}"

  if [[ "${FORCE_VOXEL_EXPORT}" == "1" ]] || [[ ! -f "${volume_raw}" ]]; then
    echo "    [1/2] Exporting voxel grid..."
    go run . \
      --input "${yaml_path}" \
      --output_dir "${vol_images}" \
      --num_projections 0 \
      --resolution "${VOLUME_RES}" \
      --export_volume \
      --transforms_file "${vol_stage}/transforms_volume_export.json" \
      --R "${CAMERA_R}" \
      --fov "${FOV_DEG}" \
      --text_progress
  else
    echo "    [1/2] Skipping voxel export (found ${volume_raw})"
  fi

  expected_size=$((VOLUME_RES * VOLUME_RES * VOLUME_RES))
  actual_size="$(wc -c <"${volume_raw}" | tr -d ' ')"
  if [[ "${actual_size}" -ne "${expected_size}" ]]; then
    echo "error: ${volume_raw} size ${actual_size} != ${expected_size}" >&2
    exit 1
  fi

  abs_raw="$(cd "${vol_stage}" && pwd)/volume.raw"
  cat >"${voxel_desc}" <<EOF
type: voxel_grid
path: ${abs_raw}
resolution: [${VOLUME_RES}, ${VOLUME_RES}, ${VOLUME_RES}]
dtype: uint8
EOF

  echo "    [2/2] Rendering projections..."
  if [[ "${stem}" == "${FIRST_STEM}" || "${stem}" == "${LAST_STEM}" ]]; then
    go run . \
      --input "${voxel_desc}" \
      --output_dir "${final_images}" \
      --num_projections "${NUM_PROJECTIONS_CANONICAL}" \
      --resolution "${RESOLUTION}" \
      --fname_pattern 'proj_%02d.png' \
      --transforms_file "${out_root}/transforms.json" \
      --R "${CAMERA_R}" \
      --fov "${FOV_DEG}"
  else
    go run . \
      --input "${voxel_desc}" \
      --output_dir "${final_images}" \
      --azimuthal_angles "${INTERMEDIATE_AZIMUTHAL_ANGLES}" \
      --polar_angles "${INTERMEDIATE_POLAR_ANGLES}" \
      --resolution "${RESOLUTION}" \
      --fname_pattern 'proj_%02d.png' \
      --transforms_file "${out_root}/transforms.json" \
      --R "${CAMERA_R}" \
      --fov "${FOV_DEG}"
  fi
done

echo "Done. Outputs under ${RENDERS_DIR}/<timestep>/"
