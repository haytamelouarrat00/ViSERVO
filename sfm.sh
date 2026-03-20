#!/usr/bin/env bash
# Run COLMAP SfM pipeline on a dataset directory.
#
# Usage: bash sfm.sh <data_dir> [name]
#   data_dir  directory containing images/
#             e.g. data/courthouse
#   name      optional base name for outputs (default: basename of data_dir)
#             e.g. courthouse  → courthouse_database.db, courthouse_sparse/
#
# Example:
#   bash sfm.sh data/courthouse
#   bash sfm.sh data/apt1/kitchen kitchen

set -euo pipefail

DATA_DIR="${1:?Usage: bash sfm.sh <data_dir> [name]}"
NAME="${2:-$(basename "$DATA_DIR")}"

DB="$DATA_DIR/${NAME}_database.db"
SPARSE="$DATA_DIR/${NAME}_sparse"

colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$DATA_DIR/images/"

colmap sequential_matcher \
    --database_path "$DB"

mkdir -p "$SPARSE"
colmap mapper \
    --database_path "$DB" \
    --image_path "$DATA_DIR/images/" \
    --output_path "$SPARSE"
