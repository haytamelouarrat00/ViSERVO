#!/usr/bin/env bash
# Split undistorted images into odd-indexed (train) and even-indexed (test).
# Run this after colmap image_undistorter has finished.
#
# Usage: bash splt.sh <dense_dir>
#   dense_dir  path containing images/ and sparse/0/
#              e.g. data/apt1/kitchen/kitchen_dense
#
# Result:
#   <dense_dir>/images_train/   symlinks to odd-indexed images  (positions 1, 3, 5, ...)
#   <dense_dir>/images_test/    symlinks to even-indexed images (positions 0, 2, 4, ...)
#
# Then train with:
#   python gaussian-splatting/train.py -s <dense_dir> --images images_train --eval

set -euo pipefail

DENSE_DIR="${1:?Usage: bash splt.sh <dense_dir>}"
IMAGES_DIR="$DENSE_DIR/images"
TRAIN_DIR="$DENSE_DIR/images_train"
TEST_DIR="$DENSE_DIR/images_test"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: $IMAGES_DIR does not exist"
    exit 1
fi

mkdir -p "$TRAIN_DIR" "$TEST_DIR"

# Remove stale symlinks from previous runs
find "$TRAIN_DIR" "$TEST_DIR" -maxdepth 1 -type l -delete

# Sort images by filename; odd positions → train, even positions → test
idx=0
while IFS= read -r -d '' img; do
    fname="$(basename "$img")"
    if (( idx % 2 == 1 )); then
        ln -sf "$(realpath --relative-to="$TRAIN_DIR" "$img")" "$TRAIN_DIR/$fname"
    else
        ln -sf "$(realpath --relative-to="$TEST_DIR"  "$img")" "$TEST_DIR/$fname"
    fi
    (( idx++ )) || true
done < <(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" \) -print0 | sort -z)

n_train=$(ls "$TRAIN_DIR" | wc -l)
n_test=$(ls  "$TEST_DIR"  | wc -l)
echo "Split complete: $n_train train (odd positions), $n_test test (even positions)"
echo "Train dir: $TRAIN_DIR"
echo "Test  dir: $TEST_DIR"
