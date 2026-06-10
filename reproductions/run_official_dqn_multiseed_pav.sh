#!/usr/bin/env bash
# Official modelfree DQN: multi-seed raw vs PAV v2, 10000 epochs (Table 7 scale).
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl4rs-tf115
pip install 'pydantic==1.10.13' -q

export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTHONPATH="/root/autodl-tmp/RL4RS-main:${PYTHONPATH:-}"
export rl4rs_benchmark_dir="/root/autodl-tmp/RL4RS-main"
export rl4rs_output_dir="${rl4rs_benchmark_dir}/output"
export rl4rs_dataset_dir="${rl4rs_benchmark_dir}/dataset"

SEEDS=(0 1 2)
EPOCHS="${EPOCHS:-10000}"
BATCH=64
LOG_EVERY="${LOG_EVERY:-500}"
PILOT_DIR="${rl4rs_output_dir}/official_multiseed"
LOG="${PILOT_DIR}/dqn_official_10000ep.log"
mkdir -p "$PILOT_DIR"

cd "${rl4rs_benchmark_dir}/script"

echo "=== Official DQN multiseed epochs=${EPOCHS} seeds=${SEEDS[*]} $(date) ===" | tee "$LOG"

for SEED in "${SEEDS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== DQN official raw seed=${SEED} epochs=${EPOCHS} $(date) ===" | tee -a "$LOG"
  python -u dqn_pav_pilot.py --seed "${SEED}" --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" --log-every "${LOG_EVERY}" 2>&1 | tee -a "$LOG"

  echo "=== DQN official pav_v2 seed=${SEED} epochs=${EPOCHS} $(date) ===" | tee -a "$LOG"
  python -u dqn_pav_pilot.py --use-pav --pav-suffix pav_v2 --seed "${SEED}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH}" --log-every "${LOG_EVERY}" 2>&1 | tee -a "$LOG"
done

echo "=== DONE $(date) ===" | tee -a "$LOG"
