#!/bin/bash
# Phase 0: env smoke (logged vs random baseline).
# Usage: bash reproductions/run_online_phase0.sh [num_episodes] [batch_size] [seed]
set -euo pipefail

script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname "$script_abs")/..
export rl4rs_output_dir="${rl4rs_benchmark_dir}/output"
export rl4rs_dataset_dir="${rl4rs_benchmark_dir}/dataset"

NUM_EPISODES=${1:-512}
BATCH_SIZE=${2:-8}
SEED=${3:-0}

cd "${rl4rs_benchmark_dir}/script"
python -u online_phase0_smoke.py \
  --num-episodes "${NUM_EPISODES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}"
