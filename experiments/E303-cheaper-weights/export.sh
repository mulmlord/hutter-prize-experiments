#!/bin/bash
# E303: a trainer checkpoint to a v4 container on the laptop, with per-group levels (train/weight_levels.patch).
#   experiments/E303-cheaper-weights/export.sh IN.tch OUTDIR [TFT_WEIGHT_LEVELS]
# Runs in builds/tftrain-wl: builds/tftrain (train/setup.sh) copied, with keep_scales.patch and
# weight_levels.patch applied (created on first use). OUTDIR/c4/weights-poc8m_w2048_pruned_v4.c4 is the result.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); LEVELS=${3:-}
T=$ROOT/builds/tftrain-wl
if [ ! -e "$T/.hp-wl-ok" ]; then
  rm -rf "$T"
  cp -a --reflink=auto "$ROOT/builds/tftrain" "$T"
  (cd "$T" && patch -p1 -s < "$ROOT/train/keep_scales.patch" && patch -p1 -s < "$ROOT/train/weight_levels.patch")
  touch "$T/.hp-wl-ok"
fi
mkdir -p "$OUT"
cd "$T"
set -a; . "$ROOT/train/env/local.env"; set +a
export TFT_EXPORT_DIR=$OUT TFT_WEIGHT_LEVELS=$LEVELS
cp "$IN" models/poc8m_w2048-pkd-q4-fp32.tch
bash run/export_chain.sh > "$OUT/export_chain.log" 2>&1
cd "$ROOT"
TREE=$T PYTHON=$T/.venv/bin/python bash neif/training/tf/4_export/export.sh \
  "$OUT/weights-poc8m_w2048.bin" "$OUT/c4" > "$OUT/export.log" 2>&1
c4=$OUT/c4/weights-poc8m_w2048_pruned_v4.c4
echo "$c4: $(stat -c %s "$c4") B, sha256 $(sha256sum "$c4" | cut -c1-16)"
