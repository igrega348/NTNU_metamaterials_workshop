#!/usr/bin/env bash
# Shortened neural_xray training for workshop Kelvin dataset (Colab T4 friendly).
# Data and checkpoints live in the workshop repo — not inside the neural_xray submodule.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NX_ROOT="${WORKSHOP_ROOT}/neural_xray"
DSET="kelvin"
DATA_DIR="${WORKSHOP_ROOT}/data/${DSET}"
OUTPUT_DIR="${WORKSHOP_ROOT}/outputs"

if [[ ! -f "${DATA_DIR}/transforms_00.json" ]]; then
  echo "error: missing ${DATA_DIR}/transforms_00.json — run stage_kelvin_for_nerf.py first" >&2
  exit 1
fi

DATA0="$(find "${DATA_DIR}" -maxdepth 1 -regex '.*/transforms_[0-9]+\.json' | sort -V | head -n 1)"
DATA1="$(find "${DATA_DIR}" -maxdepth 1 -regex '.*/transforms_[0-9]+\.json' | sort -V | tail -n 1)"
DATAALL="$(find "${DATA_DIR}" -maxdepth 1 -regex '.*/transforms_.*_to_.*\.json' | head -n 1)"
GRID0="$(find "${DATA_DIR}" -maxdepth 1 -name 'lattice_*.npz' | sort -V | head -n 1)"
GRID1="$(find "${DATA_DIR}" -maxdepth 1 -name 'lattice_*.npz' | sort -V | tail -n 1)"
if [[ -z "${GRID0}" || -z "${GRID1}" ]]; then
  echo "error: missing lattice_*.npz under ${DATA_DIR} — run stage_kelvin_for_nerf.py (converts volume.raw from renders)" >&2
  exit 1
fi

BATCH_SIZE=2048
BATCH_SIZE_VF=256
VF_NUM_SAMPLES_PER_RAY=256
NUMSTEPS=2000
DOWNSCALE_FACTOR=2
WEIGHT_NN_WIDTH=20
EVAL_BATCH_SIZE=$((BATCH_SIZE / 2))
EVAL_BATCH_SIZE_VF=$((BATCH_SIZE_VF / 2))
BSPLINE_METHOD='matrix'

echo "Workshop root: ${WORKSHOP_ROOT}"
echo "neural_xray (read-only): ${NX_ROOT}"
echo "Dataset: ${DATA_DIR}"
echo "Outputs: ${OUTPUT_DIR}"
echo "data0=${DATA0}"
echo "data1=${DATA1}"
echo "dataall=${DATAALL}"
echo "grid0=${GRID0}"
echo "grid1=${GRID1}"

# val/test splits load eval images from images_XX_<factor>/ (see multi_camera_dataparser).
echo "Creating eval downscale folders (factor ${DOWNSCALE_FACTOR})..."
bash "${SCRIPT_DIR}/resize_kelvin_for_eval.sh"

python "${NX_ROOT}/nerfstudio/nerfstudio/scripts/train.py" nerf_xray \
  --data "${DATA0}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging.local-writer.max-log-size 10 \
  --pipeline.volumetric_supervision True \
  --pipeline.volumetric_supervision_coefficient 1e-3 \
  --pipeline.datamanager.volume_grid_file "${GRID0}" \
  --pipeline.datamanager.train_num_rays_per_batch "${BATCH_SIZE}" \
  --pipeline.datamanager.eval_num_rays_per_batch "${EVAL_BATCH_SIZE}" \
  --pipeline.model.eval_num_rays_per_chunk "${EVAL_BATCH_SIZE}" \
  --pipeline.model.flat_field_trainable False \
  --max-num-iterations $((NUMSTEPS + 1)) \
  --optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
  --optimizers.fields.scheduler.lr_final 1e-4 \
  --optimizers.fields.scheduler.warmup_steps 50 \
  --optimizers.fields.scheduler.steady_steps 2000 \
  --optimizers.fields.scheduler.max_steps "${NUMSTEPS}" \
  --timestamp "canonical_F" \
  multi-camera-dataparser --downscale-factors.val "${DOWNSCALE_FACTOR}" --downscale-factors.test "${DOWNSCALE_FACTOR}"

python "${NX_ROOT}/nerfstudio/nerfstudio/scripts/train.py" nerf_xray \
  --data "${DATA1}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging.local-writer.max-log-size 10 \
  --pipeline.volumetric_supervision True \
  --pipeline.volumetric_supervision_coefficient 1e-3 \
  --pipeline.datamanager.volume_grid_file "${GRID1}" \
  --pipeline.datamanager.train_num_rays_per_batch "${BATCH_SIZE}" \
  --pipeline.datamanager.eval_num_rays_per_batch "${EVAL_BATCH_SIZE}" \
  --pipeline.model.eval_num_rays_per_chunk "${EVAL_BATCH_SIZE}" \
  --pipeline.model.flat_field_trainable False \
  --max-num-iterations $((NUMSTEPS + 1)) \
  --optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
  --optimizers.fields.scheduler.lr_final 1e-4 \
  --optimizers.fields.scheduler.warmup_steps 50 \
  --optimizers.fields.scheduler.steady_steps 2000 \
  --optimizers.fields.scheduler.max_steps "${NUMSTEPS}" \
  --timestamp "canonical_B" \
  multi-camera-dataparser --downscale-factors.val "${DOWNSCALE_FACTOR}" --downscale-factors.test "${DOWNSCALE_FACTOR}"

N1=6
STEPS="${NUMSTEPS}"
PADSTEPS=$(printf '%09d' "${STEPS}")
mkdir -p "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models"

if [[ ! -f "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models/step-${PADSTEPS}.ckpt" ]]; then
  python "${NX_ROOT}/nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py" \
    --fwd_ckpt "${OUTPUT_DIR}/${DSET}/nerf_xray/canonical_F/nerfstudio_models/step-${PADSTEPS}.ckpt" \
    --bwd_ckpt "${OUTPUT_DIR}/${DSET}/nerf_xray/canonical_B/nerfstudio_models/step-${PADSTEPS}.ckpt" \
    --out_fn "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models/step-${PADSTEPS}.ckpt"
  LOAD_OPTIMIZER=False
else
  LOAD_OPTIMIZER=True
fi

python "${NX_ROOT}/nerfstudio/nerfstudio/scripts/train.py" xray_vfield \
  --data "${DATAALL}" \
  --output_dir "${OUTPUT_DIR}" \
  --max-num-iterations "${NUMSTEPS}" \
  --steps_per_eval_image 500 \
  --steps_per_save 250 \
  --logging.local-writer.max-log-size 10 \
  --load-checkpoint "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models/step-${PADSTEPS}.ckpt" \
  --load-optimizer "${LOAD_OPTIMIZER}" \
  --pipeline.volumetric_supervision True \
  --pipeline.volumetric_supervision_coefficient 1e-4 \
  --pipeline.volumetric_supervision_start_step $((NUMSTEPS + 1000)) \
  --pipeline.datamanager.init_volume_grid_file "${GRID0}" \
  --pipeline.datamanager.final_volume_grid_file "${GRID1}" \
  --pipeline.model.deformation_field.num_control_points "${N1}" "${N1}" "${N1}" \
  --pipeline.model.deformation_field.weight_nn_width "${WEIGHT_NN_WIDTH}" \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method "${BSPLINE_METHOD}" \
  --pipeline.model.flat_field_trainable False \
  --pipeline.model.disable_mixing True \
  --pipeline.datamanager.train_num_rays_per_batch "${BATCH_SIZE_VF}" \
  --pipeline.datamanager.eval_num_rays_per_batch "${EVAL_BATCH_SIZE_VF}" \
  --pipeline.model.num_nerf_samples_per_ray "${VF_NUM_SAMPLES_PER_RAY}" \
  --timestamp "vel_${N1}" \
  --machine.seed 40 \
  multi-camera-dataparser --downscale-factors.val "${DOWNSCALE_FACTOR}" --downscale-factors.test "${DOWNSCALE_FACTOR}"

echo "Training complete. Outputs: ${OUTPUT_DIR}/${DSET}/"