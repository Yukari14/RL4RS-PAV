#!/bin/bash
# Phase 2 pilot: Q-learning raw vs PAV (PyTorch GPU + batched simulator).
# Usage: bash reproductions/run_qlearning_pilot.sh [seed] [batch_size]
set -euo pipefail

script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname "$script_abs")/..
export rl4rs_output_dir="${rl4rs_benchmark_dir}/output"
export rl4rs_dataset_dir="${rl4rs_benchmark_dir}/dataset"

SEED=${1:-0}
BATCH_SIZE=${2:-32}

cd "${rl4rs_benchmark_dir}/script"

echo "=== Q-learning raw seed=${SEED} batch=${BATCH_SIZE} (PyTorch GPU default) ==="
python -u qlearning_train.py train --seed "${SEED}" --batch-size "${BATCH_SIZE}" --num-episodes 2000

echo "=== Q-learning PAV seed=${SEED} ==="
python -u qlearning_train.py train --use-pav --seed "${SEED}" --batch-size "${BATCH_SIZE}" --num-episodes 2000
