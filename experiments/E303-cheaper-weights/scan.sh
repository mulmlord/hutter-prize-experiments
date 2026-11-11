#!/bin/bash
# E303 sensitivity scan: requantize one tensor group of t205-8q2f's container to L levels (no retraining,
# scripts/tfeval/perturb.py), then tfeval on every100 against the unchanged container. One row per group and L:
# container bytes saved, CE given up. Resumable: a finished JSON is skipped. Runs at nice 19.
#   systemd-run --user --unit e303-scan --working-directory=$PWD bash experiments/E303-cheaper-weights/scan.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/tfeval/env.sh >/dev/null
BASE=train/weights/t205-8q2f.c4
BASE_JSON=runs/tfdata/t205-8q2f-every100.json
OUT=runs/e303/scan
mkdir -p "$OUT"
TGROUPS=${TGROUPS:-"mlp.down attention.query_projection attention.key_projection attention.value_projection attention.output_projection forget_gate_projection output_gate_projection embedding"}
LEVELS=${LEVELS:-"11 7"}
for L in $LEVELS; do
  for grp in $TGROUPS; do
    tag=$(echo "$grp" | tr '.' '_')-L$L
    c4=$OUT/$tag.c4; js=$OUT/$tag-every100.json; cmp=$OUT/$tag-vs-t205.json
    [[ -s $cmp ]] && { echo "$tag: done earlier"; continue; }
    echo "$tag: start $(date -Is)"
    [[ -s $c4 ]] || python scripts/tfeval/perturb.py "$BASE" "$c4" --tensors "$grp" --levels "$L" > "$OUT/$tag.perturb.log" 2>&1 \
      || { echo "$tag: perturb FAILED"; tail -3 "$OUT/$tag.perturb.log"; continue; }
    nice -n 19 python scripts/tfeval/tfeval.py run "$c4" runs/tfdata/every100.ev --json "$js" > "$OUT/$tag.tfeval.log" 2>&1 \
      || { echo "$tag: tfeval FAILED"; tail -3 "$OUT/$tag.tfeval.log"; continue; }
    python scripts/tfeval/tfeval.py compare "$BASE_JSON" "$js" --json "$cmp" | tail -1
    echo "$tag: $(stat -c %s "$c4") B (base $(stat -c %s "$BASE") B), end $(date -Is)"
  done
done
echo "scan done $(date -Is)"
