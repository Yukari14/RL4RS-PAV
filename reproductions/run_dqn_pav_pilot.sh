#!/usr/bin/env bash
# Small RLlib DQN pilot: raw vs PAV (seed 0). Uses local SlateRecEnv + GPU TF in rl4rs-tf115.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl4rs-tf115

export TMPDIR=/root/autodl-tmp/tmp
export rl4rs_benchmark_dir="$(cd "$(dirname "$0")/.." && pwd)"
export rl4rs_output_dir="${rl4rs_benchmark_dir}/output"
export rl4rs_dataset_dir="${rl4rs_benchmark_dir}/dataset"

EPOCHS="${EPOCHS:-500}"
BATCH="${BATCH:-64}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-50}"

cd "${rl4rs_benchmark_dir}/script"

echo "========== DQN raw seed=${SEED} epochs=${EPOCHS} =========="
python -u dqn_pav_pilot.py \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --log-every "${LOG_EVERY}"

echo "========== DQN + PAV seed=${SEED} epochs=${EPOCHS} =========="
python -u dqn_pav_pilot.py \
  --use-pav \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --log-every "${LOG_EVERY}"

echo "Done. Summaries in ${rl4rs_output_dir}/dqn_pilot/"
