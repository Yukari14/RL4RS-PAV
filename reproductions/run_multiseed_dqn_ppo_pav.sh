#!/usr/bin/env bash
# Multi-seed online raw vs PAV v2: DQN (pilot) + official PPO, 100 epochs each.
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
EPOCHS=100
BATCH=64
LOG_EVERY=25
PILOT_DIR="${rl4rs_output_dir}/multiseed_pilot"
LOG="${PILOT_DIR}/multiseed_dqn_ppo_100ep.log"
mkdir -p "$PILOT_DIR"

cd "${rl4rs_benchmark_dir}/script"

echo "=== Multi-seed DQN+PPO raw vs pav_v2 epochs=${EPOCHS} seeds=${SEEDS[*]} $(date) ===" | tee "$LOG"

for SEED in "${SEEDS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== DQN raw seed=${SEED} $(date) ===" | tee -a "$LOG"
  python -u dqn_pav_pilot.py --seed "${SEED}" --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" --log-every "${LOG_EVERY}" 2>&1 | tee -a "$LOG"

  echo "=== DQN pav_v2 seed=${SEED} $(date) ===" | tee -a "$LOG"
  python -u dqn_pav_pilot.py --use-pav --pav-suffix pav_v2 --seed "${SEED}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH}" --log-every "${LOG_EVERY}" 2>&1 | tee -a "$LOG"
done

for SEED in "${SEEDS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== Official PPO raw seed=${SEED} $(date) ===" | tee -a "$LOG"
  python -u ppo_pav_pilot.py --seed "${SEED}" --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" --log-every "${LOG_EVERY}" 2>&1 | tee -a "$LOG"

  echo "=== Official PPO pav_v2 seed=${SEED} $(date) ===" | tee -a "$LOG"
  python -u ppo_pav_pilot.py --use-pav --pav-suffix pav_v2 --seed "${SEED}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH}" --log-every "${LOG_EVERY}" 2>&1 | tee -a "$LOG"
done

echo "=== DONE $(date) ===" | tee -a "$LOG"
echo "Summaries:" | tee -a "$LOG"
ls -1 "${rl4rs_output_dir}/dqn_pilot"/dqn_*_seed*_summary.json 2>/dev/null | tee -a "$LOG" || true
ls -1 "${rl4rs_output_dir}/ppo_pilot"/ppo_official_*_seed*_summary.json 2>/dev/null | tee -a "$LOG" || true
