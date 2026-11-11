#!/bin/bash
# E303: every100 tfeval of exported containers against t205-8q2f, at nice 19.
#   experiments/E303-cheaper-weights/evalwarm.sh DIR...   (each DIR from export.sh; DIR/c4/*_pruned_v4.c4)
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/tfeval/env.sh >/dev/null
for d in "$@"; do
  c4=$d/c4/weights-poc8m_w2048_pruned_v4.c4
  while [ ! -s "$c4" ]; do sleep 60; done
  [ -s "$d/vs-t205.json" ] && { echo "$d: done earlier"; continue; }
  echo "$d: start $(date -Is)"
  nice -n 19 python scripts/tfeval/tfeval.py run "$c4" runs/tfdata/every100.ev --json "$d/every100.json" > "$d/tfeval.log" 2>&1 \
    || { echo "$d: tfeval FAILED"; tail -3 "$d/tfeval.log"; continue; }
  python scripts/tfeval/tfeval.py compare runs/tfdata/t205-8q2f-every100.json "$d/every100.json" --json "$d/vs-t205.json" | tail -1
  echo "$d: $(stat -c %s "$c4") B (t205-8q2f 3715490 B), end $(date -Is)"
done
echo "evalwarm done $(date -Is)"
