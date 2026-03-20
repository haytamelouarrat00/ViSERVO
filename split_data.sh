#!/usr/bin/env bash
# Usage: bash split_data.sh <data_dir>
#   e.g. bash split_data.sh data/apt2/luke

DIR="${1:?Usage: bash split_data.sh <data_dir>}"

mkdir -p "$DIR/images" "$DIR/poses" "$DIR/depths"

for f in "$DIR/data"/*.color.jpg; do ln -sf "$(realpath "$f")" "$DIR/images/$(basename "$f" .color.jpg).jpg"; done
for f in "$DIR/data"/*.pose.txt;  do ln -sf "$(realpath "$f")" "$DIR/poses/$(basename "$f")"; done
for f in "$DIR/data"/*.depth.png; do ln -sf "$(realpath "$f")" "$DIR/depths/$(basename "$f")"; done