#!/usr/bin/env bash
# Two-stage Kelvin-indent X-ray pipeline using neural_xray/xray_projection_render:
#
#   1) Analytical YAML -> UINT8 voxel grid (volume.raw) via --export_volume and
#      --num_projections 0 (no direct projections from the mesh collection).
#   2) voxel_grid descriptor -> projections by ray-marching the loaded grid
#      (same renderer path, density = trilinear sample of the volume).
#
# volume.raw layout matches the engine: ZXY order, UINT8 normalized to max density
# in the sampled cube [-1,1]^3 (see xray_projection_render README / main.go).
#
# Reproducibility: 32 in-plane equispaced views (no --out_of_plane). Same Go/OS/arch
# for stable floats. Voxel values depend on VOLUME_RES and normalization inside export.
#
# Outputs under ./renders/<stem>/:
#   volume_stage/volume.raw, volume_stage/voxel_grid.yaml (and stage-1 object.json)
#   images/proj_XX.png, transforms.json, object.json (voxel_grid metadata)
#
# Environment:
#   YAML_GLOB        basename glob under this directory (default: kelvin_indent_t*.yaml)
#                    e.g. YAML_GLOB='renderer_regen2_*_t*.yaml' ./render_files.sh
#   VOLUME_RES       voxel grid edge length (default 128; memory = VOLUME_RES^3 bytes)
#   RESOLUTION       projection image size (default 512)
#   NUM_PROJECTIONS  (default 32)
#   CAMERA_R, FOV_DEG — passed to both stages where applicable
#   FORCE_VOXEL_EXPORT=1  re-run stage 1 even if volume.raw already exists
#
# Options:
#   --clear  remove kelvin_indent/renders/ (all voxel exports and projections), then exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XRAY_DIR="${REPO_ROOT}/neural_xray/xray_projection_render"
RENDERS_DIR="${SCRIPT_DIR}/renders"

usage() {
  echo "usage: ${0##*/} [--clear]" >&2
  echo "  (no args)  run the two-stage render pipeline" >&2
  echo "  --clear    delete ${RENDERS_DIR} and exit" >&2
  echo "Set YAML_GLOB to choose inputs (default: kelvin_indent_t*.yaml)." >&2
}

do_clear=0
for arg in "$@"; do
  case "${arg}" in
    --clear)
      do_clear=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
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
    echo "Nothing to remove (${RENDERS_DIR} is missing or not a directory)"
  fi
  exit 0
fi

VOLUME_RES="${VOLUME_RES:-128}"
RESOLUTION="${RESOLUTION:-512}"
NUM_PROJECTIONS="${NUM_PROJECTIONS:-32}"
CAMERA_R="${CAMERA_R:-4}"
FOV_DEG="${FOV_DEG:-40}"
FORCE_VOXEL_EXPORT="${FORCE_VOXEL_EXPORT:-0}"
YAML_GLOB="${YAML_GLOB:-kelvin_indent_t*.yaml}"

if [[ ! -d "${XRAY_DIR}" ]]; then
  echo "error: xray_projection_render not found at ${XRAY_DIR}" >&2
  exit 1
fi

shopt -s nullglob
yaml_files=("${SCRIPT_DIR}"/${YAML_GLOB})
shopt -u nullglob

if [[ ${#yaml_files[@]} -eq 0 ]]; then
  echo "error: no files matching ${YAML_GLOB} under ${SCRIPT_DIR}" >&2
  exit 1
fi

IFS=$'\n' yaml_files=($(sort <<<"${yaml_files[*]}"))
unset IFS

echo "Pipeline (${YAML_GLOB}): YAML -> volume.raw (${VOLUME_RES}^3) -> ${NUM_PROJECTIONS} projections @ ${RESOLUTION}px"
echo "Using xray_projection_render from: ${XRAY_DIR}"

cd "${XRAY_DIR}"

for yaml_path in "${yaml_files[@]}"; do
  stem="$(basename "${yaml_path}" .yaml)"
  out_root="${SCRIPT_DIR}/renders/${stem}"
  vol_stage="${out_root}/volume_stage"
  vol_images="${vol_stage}/images"
  final_images="${out_root}/images"
  volume_raw="${vol_stage}/volume.raw"
  voxel_desc="${vol_stage}/voxel_grid.yaml"

  mkdir -p "${vol_images}" "${final_images}"

  echo "==> ${stem}"

  if [[ "${FORCE_VOXEL_EXPORT}" == "1" ]] || [[ ! -f "${volume_raw}" ]]; then
    echo "    [1/2] Exporting voxel grid from analytical YAML..."
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
    echo "    [1/2] Skipping voxel export (found ${volume_raw}; set FORCE_VOXEL_EXPORT=1 to rebuild)"
  fi

  expected_size=$((VOLUME_RES * VOLUME_RES * VOLUME_RES))
  actual_size="$(wc -c <"${volume_raw}" | tr -d ' ')"
  if [[ "${actual_size}" -ne "${expected_size}" ]]; then
    echo "error: ${volume_raw} size ${actual_size} != ${expected_size} (wrong VOLUME_RES?)" >&2
    exit 1
  fi

  # Absolute path: loader runs with cwd = xray_projection_render.
  abs_raw="$(cd "${vol_stage}" && pwd)/volume.raw"
  cat >"${voxel_desc}" <<EOF
type: voxel_grid
path: ${abs_raw}
resolution: [${VOLUME_RES}, ${VOLUME_RES}, ${VOLUME_RES}]
dtype: uint8
EOF

  echo "    [2/2] Rendering ${NUM_PROJECTIONS} projections from voxel grid..."
  go run . \
    --input "${voxel_desc}" \
    --output_dir "${final_images}" \
    --num_projections "${NUM_PROJECTIONS}" \
    --resolution "${RESOLUTION}" \
    --fname_pattern 'proj_%02d.png' \
    --transforms_file "${out_root}/transforms.json" \
    --R "${CAMERA_R}" \
    --fov "${FOV_DEG}"
done

echo "Done. Outputs under ${SCRIPT_DIR}/renders/<timestep>/"
