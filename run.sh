#!/usr/bin/env bash
# run.sh – Build the pipeline, generate test data, and execute.
# Usage: ./run.sh [--count N] [--width W] [--height H]
set -euo pipefail

COUNT=200
WIDTH=512
HEIGHT=512
INPUT_DIR="data/input"
OUTPUT_DIR="output"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --count)  COUNT="$2";  shift 2 ;;
    --width)  WIDTH="$2";  shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== GPU Image Pipeline ==="
echo "  Images : $COUNT"
echo "  Size   : ${WIDTH}x${HEIGHT}"
echo ""

# 1. Generate synthetic test images if the input dir is empty
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
if [ -z "$(ls -A "$INPUT_DIR"/*.pgm 2>/dev/null)" ]; then
  echo "[1/3] Generating $COUNT synthetic PGM images..."
  python3 scripts/gen_pgm.py \
    --count "$COUNT" --width "$WIDTH" --height "$HEIGHT" \
    --output "$INPUT_DIR"
else
  echo "[1/3] Input images already present – skipping generation."
fi

# 2. Build
echo "[2/3] Building..."
make -j"$(nproc)"

# 3. Run pipeline
echo "[3/3] Running pipeline..."
LOG_FILE="execution_log.txt"
./bin/image_pipeline \
    --input  "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done! Outputs are in: $OUTPUT_DIR"
echo "Log saved to       : $LOG_FILE"
