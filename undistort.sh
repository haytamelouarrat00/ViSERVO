#!/usr/bin/env bash
# Run COLMAP image undistortion and optionally train a Gaussian Splatting scene.
#
# Usage: bash undistort.sh <data_dir> [name] [resolution]
#   data_dir    directory containing images/ and <name>_sparse/
#               e.g. data/courthouse
#   name        optional base name (default: basename of data_dir)
#               e.g. courthouse  → courthouse_sparse/, courthouse_dense/, courthouse_gs/
#   resolution  optional downscale factor for 3DGS training (default: 4)
#
# Example:
#   bash undistort.sh data/courthouse
#   bash undistort.sh data/apt1/kitchen kitchen 2

set -euo pipefail

DATA_DIR="${1:?Usage: bash undistort.sh <data_dir> [name] [resolution]}"
NAME="${2:-$(basename "$DATA_DIR")}"
RESOLUTION="${3:-4}"

SPARSE="$DATA_DIR/${NAME}_sparse/0"
DENSE="$DATA_DIR/${NAME}_dense"
GS_MODEL="$DATA_DIR/${NAME}_gs"

colmap image_undistorter \
    --image_path "$DATA_DIR/images" \
    --input_path "$SPARSE" \
    --output_path "$DENSE" \
    --output_type COLMAP

mkdir -p "$DENSE/sparse/0"
mv "$DENSE/sparse/"*.bin "$DENSE/sparse/0/"

python gaussian-splatting/train.py \
    -s "$DENSE" \
    -m "$GS_MODEL" \
    --eval \
    --resolution "$RESOLUTION"
