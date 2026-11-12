#!/bin/bash
# E310: a short bf16 smoke of TFT_HIER (train/hier.py) on the laptop GPU, through train/smoke_phase.py, in
# builds/tftrain-rate (builds/tftrain + keep_scales, weight_levels, rate_term and hier patches).
#   HIER="2-13" TAG=hier experiments/E310-hierarchical-transformer/smoke.sh      HIER empty: the plain model
# Both arms train from the same initialisation on the same documents in the same order.
set -euo pipefail
cd "$(dirname "$0")/../../builds/tftrain-rate"
set -a; . ../../train/env/local.env; set +a
cp ../../train/hier.py tft/hier.py
L=../../runs/e310/smoke/logs-$TAG
rm -rf "$L"; mkdir -p "$L"
unset TFT_RATE_LAMBDA TFT_WEIGHT_LEVELS TFT_WARMSTART
export TFT_HIER="${HIER:-}" HP_SMOKE_EVERY=${EVERY:-50} HP_SMOKE_OFFSET=10
export TFT_MICRO_BATCH_TOKENS=${MB:-32768} TFT_OPTIMIZER_BATCH_TOKENS=${OB:-262144} TFT_EPOCHS_BF16=${EPOCHS:-2}
export TFT_LOG_DIR=$L TFT_EXPORT_DIR=../../runs/e310/smoke/export TFT_RUN_TAG=smoke-$TAG
echo "smoke $TAG: TFT_HIER='$TFT_HIER', start $(date -Is)"
.venv/bin/torchrun --nproc_per_node 1 --master_port 29620 ../../train/smoke_phase.py bf16
echo "smoke $TAG done $(date -Is)"
