#!/usr/bin/env bash
# Dynamic GPU queue launcher for stage_train (one run per GPU at a time).
#
# Usage:
#   scripts/launch_relay_eps.sh "0 1 2 3"          # GPU ids
#   scripts/launch_relay_eps.sh "0 1" --smoke      # pipeline check
#
# Restart-safe: stage_train skips runs whose weights.npz already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:experiments/relay_eps:${PYTHONPATH:-}"

GPUS=(${1:-0})
shift || true
EXTRA_ARGS=("$@")

# Forward --config to run_matrix.py so the queue enumerates the SAME
# matrix that stage_train will accept (bug fix: previously the queue was
# always built from the default config).
CFG="configs/relay_eps.yaml"
prev=""
for a in "${EXTRA_ARGS[@]}"; do
    if [[ "$prev" == "--config" ]]; then CFG="$a"; fi
    if [[ "$a" == --config=* ]]; then CFG="${a#--config=}"; fi
    prev="$a"
done

mapfile -t RUNS < <(python3 experiments/relay_eps/run_matrix.py --list --config "$CFG")
echo "queued ${#RUNS[@]} runs on GPUs: ${GPUS[*]}"

QUEUE_FILE=$(mktemp)
printf "%s\n" "${RUNS[@]}" > "$QUEUE_FILE"

worker() {
  local gpu=$1
  while true; do
    local run
    run=$(flock "$QUEUE_FILE.lock" bash -c \
      "head -n1 '$QUEUE_FILE' && sed -i '1d' '$QUEUE_FILE'") || true
    [[ -z "$run" ]] && break
    echo "[gpu $gpu] -> $run"
    CUDA_VISIBLE_DEVICES=$gpu python3 experiments/relay_eps/stage_train.py \
      --run-id "$run" "${EXTRA_ARGS[@]}" \
      > "results/relay_eps/logs/${run}.log" 2>&1 || \
      echo "[gpu $gpu] FAILED: $run (see log)"
  done
}

mkdir -p results/relay_eps/logs
touch "$QUEUE_FILE.lock"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
rm -f "$QUEUE_FILE" "$QUEUE_FILE.lock"
echo "all runs finished."
