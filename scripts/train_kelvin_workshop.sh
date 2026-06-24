#!/usr/bin/env bash
# Shortened neural_xray training for workshop Kelvin dataset (Colab T4 friendly).
# Data and checkpoints live in the workshop repo — not inside the neural_xray submodule.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NX_ROOT="${WORKSHOP_ROOT}/neural_xray"

# Select dataset: DATASET=indentation (default) or DATASET=uniform
DATASET="${DATASET:-indentation}"
if [[ "${DATASET}" == "uniform" ]]; then
  DATA_DIR="${WORKSHOP_ROOT}/data/kelvin_uniform"
elif [[ "${DATASET}" == "indentation" ]]; then
  DATA_DIR="${WORKSHOP_ROOT}/data/kelvin"
else
  echo "error: unknown DATASET '${DATASET}' — use 'indentation' or 'uniform'" >&2
  exit 1
fi
echo "DATASET: ${DATASET} (${DATA_DIR})"

DSET="${DSET:-kelvin_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
echo "DSET: ${DSET}"
OUTPUT_DIR="${WORKSHOP_ROOT}/outputs"

if [[ ! -f "${DATA_DIR}/transforms_00.json" ]]; then
  echo "error: missing ${DATA_DIR}/transforms_00.json — run stage_kelvin_for_nerf.py first" >&2
  exit 1
fi

DATA0="$(find "${DATA_DIR}" -maxdepth 1 -regex '.*/transforms_[0-9]+\.json' | sort -V | head -n 1)"
DATA1="$(find "${DATA_DIR}" -maxdepth 1 -regex '.*/transforms_[0-9]+\.json' | sort -V | tail -n 1)"
DATAALL="$(find "${DATA_DIR}" -maxdepth 1 -regex '.*/transforms_.*_to_.*\.json' | sort -V | tail -n 1)"
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
NUMSTEPS_VF6=6000
NUMSTEPS_VF9=4000
NUMSTEPS_MIX=3000
DOWNSCALE_FACTOR=4
WEIGHT_NN_WIDTH=20
EVAL_BATCH_SIZE=$((BATCH_SIZE / 2))
EVAL_BATCH_SIZE_VF=$((BATCH_SIZE_VF / 2))
BSPLINE_METHOD='matrix'

if [[ -n "${DEMO_FAST:-}" ]]; then
  NUMSTEPS=500
  NUMSTEPS_VF6=500
  NUMSTEPS_VF9=500
  NUMSTEPS_MIX=500
  echo "DEMO_FAST mode: all stages capped at 500 steps"
fi

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
DATA_DIR="${DATA_DIR}" bash "${SCRIPT_DIR}/resize_kelvin_for_eval.sh"

PADSTEPS_CANONICAL=$(printf '%09d' "${NUMSTEPS}")
CKPT_F="${OUTPUT_DIR}/${DSET}/nerf_xray/canonical_F/nerfstudio_models/step-${PADSTEPS_CANONICAL}.ckpt"
CKPT_B="${OUTPUT_DIR}/${DSET}/nerf_xray/canonical_B/nerfstudio_models/step-${PADSTEPS_CANONICAL}.ckpt"

if [[ -f "${CKPT_F}" ]]; then
  echo "Skipping canonical_F training — checkpoint exists: ${CKPT_F}"
else
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
fi

if [[ -f "${CKPT_B}" ]]; then
  echo "Skipping canonical_B training — checkpoint exists: ${CKPT_B}"
else
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
fi

VFIELD_RES_6_LRPW=1e-3
VFIELD_RES_6_WUS=3000
VFIELD_RES_9_LRPW=1e-3
VFIELD_RES_9_WUS=1000
N_MIX_FIELD=6

# --- vel_6 ---
N1=6
STEPS_VF6="${NUMSTEPS}"       # canonical checkpoint step (input to combine)
PADSTEPS_VF6=$(printf '%09d' "${STEPS_VF6}")
mkdir -p "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models"

if [[ ! -f "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models/step-${PADSTEPS_VF6}.ckpt" ]]; then
  python "${NX_ROOT}/nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py" \
    --fwd_ckpt "${OUTPUT_DIR}/${DSET}/nerf_xray/canonical_F/nerfstudio_models/step-${PADSTEPS_VF6}.ckpt" \
    --bwd_ckpt "${OUTPUT_DIR}/${DSET}/nerf_xray/canonical_B/nerfstudio_models/step-${PADSTEPS_VF6}.ckpt" \
    --out_fn "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models/step-${PADSTEPS_VF6}.ckpt"
  LOAD_OPTIMIZER_VF6=False
else
  LOAD_OPTIMIZER_VF6=True
fi

python "${NX_ROOT}/nerfstudio/nerfstudio/scripts/train.py" xray_vfield \
  --data "${DATAALL}" \
  --output_dir "${OUTPUT_DIR}" \
  --max-num-iterations "${NUMSTEPS_VF6}" \
  --steps_per_eval_image 500 \
  --steps_per_save 250 \
  --logging.local-writer.max-log-size 10 \
  --load-checkpoint "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/nerfstudio_models/step-${PADSTEPS_VF6}.ckpt" \
  --load-optimizer "${LOAD_OPTIMIZER_VF6}" \
  --pipeline.volumetric_supervision True \
  --pipeline.volumetric_supervision_coefficient 1e-4 \
  --pipeline.volumetric_supervision_start_step $((NUMSTEPS + 1000)) \
  --pipeline.datamanager.init_volume_grid_file "${GRID0}" \
  --pipeline.datamanager.final_volume_grid_file "${GRID1}" \
  --pipeline.model.deformation_field.num_control_points "${N1}" "${N1}" "${N1}" \
  --pipeline.model.deformation_field.weight_nn_width "${WEIGHT_NN_WIDTH}" \
  --pipeline.model.deformation_field.weight_nn_gain 2.0 \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method "${BSPLINE_METHOD}" \
  --pipeline.model.flat_field_trainable False \
  --pipeline.model.train_field_weighing False \
  --pipeline.model.disable_mixing True \
  --pipeline.datamanager.train_num_rays_per_batch "${BATCH_SIZE_VF}" \
  --pipeline.datamanager.eval_num_rays_per_batch "${EVAL_BATCH_SIZE_VF}" \
  --pipeline.model.eval_num_rays_per_chunk "${EVAL_BATCH_SIZE_VF}" \
  --pipeline.model.num_nerf_samples_per_ray "${VF_NUM_SAMPLES_PER_RAY}" \
  --optimizers.fields.optimizer.lr 5e-4 \
  --optimizers.fields.optimizer.weight_decay 1e-1 \
  --optimizers.fields.scheduler.lr_pre_warmup "${VFIELD_RES_6_LRPW}" \
  --optimizers.fields.scheduler.lr_final 1e-6 \
  --optimizers.fields.scheduler.warmup_steps "${VFIELD_RES_6_WUS}" \
  --optimizers.fields.scheduler.steady_steps $((NUMSTEPS_VF6 - 1000)) \
  --optimizers.fields.scheduler.max_steps "${NUMSTEPS_VF6}" \
  --timestamp "vel_${N1}" \
  --machine.seed 40 \
  multi-camera-dataparser --downscale-factors.val "${DOWNSCALE_FACTOR}" --downscale-factors.test "${DOWNSCALE_FACTOR}"

# --- vel_9 ---
N2=9
STEPS_VF9_INIT=$((NUMSTEPS + NUMSTEPS_VF6))              # step label of the refine_vfield output (canonical + vel_6)
STEPS_VF9_FINAL=$((STEPS_VF9_INIT + NUMSTEPS_VF9))       # step label after vel_9 training completes
PADSTEPS_VF9_INIT=$(printf '%09d' "${STEPS_VF9_INIT}")
PADSTEPS_VF9_FINAL=$(printf '%09d' "${STEPS_VF9_FINAL}")
mkdir -p "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N2}/nerfstudio_models"

if [[ ! -f "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N2}/nerfstudio_models/step-${PADSTEPS_VF9_INIT}.ckpt" ]]; then
  python "${NX_ROOT}/nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py" \
    --load-config "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N1}/config.yml" \
    --new-resolution "${N2}" \
    --new-nn-width "${WEIGHT_NN_WIDTH}" \
    --out-path "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N2}/nerfstudio_models/step-${PADSTEPS_VF9_INIT}.ckpt"
  LOAD_OPTIMIZER_VF9=False
else
  LOAD_OPTIMIZER_VF9=False
fi

python "${NX_ROOT}/nerfstudio/nerfstudio/scripts/train.py" xray_vfield \
  --data "${DATAALL}" \
  --output_dir "${OUTPUT_DIR}" \
  --max-num-iterations "${NUMSTEPS_VF9}" \
  --steps_per_eval_image 500 \
  --steps_per_save 250 \
  --logging.local-writer.max-log-size 10 \
  --load-checkpoint "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N2}/nerfstudio_models/step-${PADSTEPS_VF9_INIT}.ckpt" \
  --load-optimizer "${LOAD_OPTIMIZER_VF9}" \
  --pipeline.volumetric_supervision True \
  --pipeline.volumetric_supervision_coefficient 1e-4 \
  --pipeline.datamanager.init_volume_grid_file "${GRID0}" \
  --pipeline.datamanager.final_volume_grid_file "${GRID1}" \
  --pipeline.model.deformation_field.num_control_points "${N2}" "${N2}" "${N2}" \
  --pipeline.model.deformation_field.weight_nn_width "${WEIGHT_NN_WIDTH}" \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method "${BSPLINE_METHOD}" \
  --pipeline.model.flat_field_trainable False \
  --pipeline.model.train_field_weighing False \
  --pipeline.model.disable_mixing True \
  --pipeline.datamanager.train_num_rays_per_batch "${BATCH_SIZE_VF}" \
  --pipeline.datamanager.eval_num_rays_per_batch "${EVAL_BATCH_SIZE_VF}" \
  --pipeline.model.eval_num_rays_per_chunk "${EVAL_BATCH_SIZE_VF}" \
  --pipeline.model.num_nerf_samples_per_ray "${VF_NUM_SAMPLES_PER_RAY}" \
  --optimizers.fields.optimizer.lr 1e-4 \
  --optimizers.fields.optimizer.weight_decay 1e-1 \
  --optimizers.fields.scheduler.lr_pre_warmup "${VFIELD_RES_9_LRPW}" \
  --optimizers.fields.scheduler.lr_final 1e-6 \
  --optimizers.fields.scheduler.warmup_steps "${VFIELD_RES_9_WUS}" \
  --optimizers.fields.scheduler.steady_steps $((NUMSTEPS_VF9 - 1000)) \
  --optimizers.fields.scheduler.max_steps "${NUMSTEPS_VF9}" \
  --timestamp "vel_${N2}" \
  --machine.seed 40 \
  multi-camera-dataparser --downscale-factors.val "${DOWNSCALE_FACTOR}" --downscale-factors.test "${DOWNSCALE_FACTOR}"

# --- spatiotemporal_mix ---
mkdir -p "${OUTPUT_DIR}/${DSET}/spatiotemporal_mix/vel_${N2}/nerfstudio_models"
cp "${OUTPUT_DIR}/${DSET}/xray_vfield/vel_${N2}/nerfstudio_models/step-${PADSTEPS_VF9_FINAL}.ckpt" \
   "${OUTPUT_DIR}/${DSET}/spatiotemporal_mix/vel_${N2}/nerfstudio_models/step-${PADSTEPS_VF9_FINAL}.ckpt"

python "${NX_ROOT}/nerfstudio/nerfstudio/scripts/train.py" spatiotemporal_mix \
  --data "${DATAALL}" \
  --output_dir "${OUTPUT_DIR}" \
  --max-num-iterations "${NUMSTEPS_MIX}" \
  --steps_per_eval_image 500 \
  --steps_per_save 250 \
  --logging.local-writer.max-log-size 10 \
  --load-checkpoint "${OUTPUT_DIR}/${DSET}/spatiotemporal_mix/vel_${N2}/nerfstudio_models/step-${PADSTEPS_VF9_FINAL}.ckpt" \
  --load-optimizer False \
  --pipeline.volumetric_supervision False \
  --pipeline.datamanager.init_volume_grid_file "${GRID0}" \
  --pipeline.datamanager.final_volume_grid_file "${GRID1}" \
  --pipeline.model.field_weighing.num_control_points "${N_MIX_FIELD}" "${N_MIX_FIELD}" "${N_MIX_FIELD}" \
  --pipeline.model.deformation_field.num_control_points "${N2}" "${N2}" "${N2}" \
  --pipeline.model.deformation_field.weight_nn_width "${WEIGHT_NN_WIDTH}" \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method "${BSPLINE_METHOD}" \
  --pipeline.model.flat_field_trainable False \
  --pipeline.model.train_field_weighing True \
  --pipeline.model.disable_mixing False \
  --pipeline.datamanager.train_num_rays_per_batch "${BATCH_SIZE_VF}" \
  --pipeline.datamanager.eval_num_rays_per_batch "${EVAL_BATCH_SIZE_VF}" \
  --pipeline.model.eval_num_rays_per_chunk "${EVAL_BATCH_SIZE_VF}" \
  --pipeline.model.num_nerf_samples_per_ray "${VF_NUM_SAMPLES_PER_RAY}" \
  --optimizers.field_weighing.optimizer.lr 1e-2 \
  --optimizers.field_weighing.optimizer.weight_decay 1e-1 \
  --optimizers.field_weighing.scheduler.steady_steps "${NUMSTEPS_MIX}" \
  --optimizers.field_weighing.scheduler.max_steps "${NUMSTEPS_MIX}" \
  --optimizers.field_weighing.scheduler.warmup_steps 200 \
  --timestamp "vel_${N2}" \
  --machine.seed 40 \
  multi-camera-dataparser --downscale-factors.val "${DOWNSCALE_FACTOR}" --downscale-factors.test "${DOWNSCALE_FACTOR}"

echo "Training complete. Outputs: ${OUTPUT_DIR}/${DSET}/"