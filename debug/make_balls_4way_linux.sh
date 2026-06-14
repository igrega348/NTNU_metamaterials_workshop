#!/bin/bash
# Adapted from make_balls_4way.sh for Linux/CUDA machines.
# Generates the 4-way in-plane / out-of-plane dataset used by check_oop.sh.
# Outputs to neural_xray/data/simulated/balls_4way/ (matching run_canonical.sh PROJECT_ROOT).
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDERER="$REPO_ROOT/neural_xray/xray_projection_render/xray_render_cuda"
INPUT="$REPO_ROOT/debug/balls.json"
DATADIR="$REPO_ROOT/neural_xray/data/simulated/balls_4way"
COMBINE_PY="$REPO_ROOT/neural_xray/nerf_data/scripts/combine_transforms.py"
RAW2NPY_PY="$REPO_ROOT/neural_xray/nerf_data/scripts/raw_to_npy.py"

RES=500
RES_VOL=250
NUM_TRAIN=32
NUM_EVAL=5

mkdir -p "$DATADIR"
cd "$DATADIR"

# Export volume (volume.raw and object.json land in DATADIR since output_dir defaults to "images"
# and filepath.Dir("images") = ".")
echo "=== Exporting volume grid (${RES_VOL}^3) ==="
"$RENDERER" --input "$INPUT" --resolution ${RES_VOL} --num_projections 0 --export_volume

# Render training images
echo "=== Rendering in-plane training images (${NUM_TRAIN} @ ${RES}px) ==="
"$RENDERER" --input "$INPUT" --resolution $RES --num_projections $NUM_TRAIN \
  --fname_pattern 'train_%03d.png' --transforms_file 'transforms_0.json' \
  --output_dir 'images_inplane'

echo "=== Rendering out-of-plane training images (${NUM_TRAIN} @ ${RES}px) ==="
"$RENDERER" --input "$INPUT" --resolution $RES --num_projections $NUM_TRAIN \
  --fname_pattern 'train_%03d.png' --transforms_file 'transforms_1.json' \
  --out_of_plane --output_dir 'images_outplane'

# Render eval images into the same image dirs (different filename pattern)
echo "=== Rendering in-plane eval images (${NUM_EVAL} @ ${RES}px) ==="
"$RENDERER" --input "$INPUT" --resolution $RES --num_projections $NUM_EVAL \
  --fname_pattern 'eval_%03d.png' --transforms_file 'transforms_2.json' \
  --output_dir 'images_inplane'

echo "=== Rendering out-of-plane eval images (${NUM_EVAL} @ ${RES}px) ==="
"$RENDERER" --input "$INPUT" --resolution $RES --num_projections $NUM_EVAL \
  --fname_pattern 'eval_%03d.png' --transforms_file 'transforms_3.json' \
  --out_of_plane --output_dir 'images_outplane'

# Combine train+eval transforms into the 4 experiment configs
echo "=== Combining transforms ==="
mkdir -p temp

cp transforms_0.json transforms_2.json ./temp/
python3 "$COMBINE_PY" --folder ./temp --timestamp-func 'lambda x: 0.0' --no-enforce_exists \
  && mv ./temp/transforms.json ./transforms_in_in.json
rm ./temp/*

cp transforms_0.json transforms_3.json ./temp/
python3 "$COMBINE_PY" --folder ./temp --timestamp-func 'lambda x: 0.0' --no-enforce_exists \
  && mv ./temp/transforms.json ./transforms_in_out.json
rm ./temp/*

cp transforms_1.json transforms_2.json ./temp/
python3 "$COMBINE_PY" --folder ./temp --timestamp-func 'lambda x: 0.0' --no-enforce_exists \
  && mv ./temp/transforms.json ./transforms_out_in.json
rm ./temp/*

cp transforms_1.json transforms_3.json ./temp/
python3 "$COMBINE_PY" --folder ./temp --timestamp-func 'lambda x: 0.0' --no-enforce_exists \
  && mv ./temp/transforms.json ./transforms_out_out.json
rm -r ./temp

# Convert volume.raw to NPZ for normed_correlation target (if needed as VoxelGrid)
echo "=== Converting volume.raw to NPZ ==="
python3 "$RAW2NPY_PY" --input volume.raw --dtype UINT8 --resolution $RES_VOL $RES_VOL $RES_VOL

echo ""
echo "=== Data generation complete ==="
echo "Dataset: $DATADIR"
ls -la "$DATADIR"
