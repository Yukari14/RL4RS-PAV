# Online RL + PAV 测试手册

明天在 **有 GPU、内存 ≥16GB** 的机器上按顺序执行。当前无卡 2GB 环境无法跑 DIEN simulator。

## 环境准备

**推荐 GPU 环境 `rl4rs-tf115`**（Python 3.8 + NVIDIA TF 1.15.5+nv23.02 + PyTorch 2.4 cu118，支持 RTX 4090 sm_89）：

```bash
# 首次安装（有 wget/pip 进度条，大文件在 /root/autodl-tmp/wheels）
bash reproductions/setup_rl4rs_tf115_gpu.sh
# 实时看进度：tail -f /root/autodl-tmp/setup_gpu.log

conda activate rl4rs-tf115
export TMPDIR=/root/autodl-tmp/tmp PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
cd /root/autodl-tmp/RL4RS-main
export rl4rs_output_dir=$PWD/output
export rl4rs_dataset_dir=$PWD/dataset
```

验证：`python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"` 和 `python -c "import torch; print(torch.cuda.is_available())"`

旧环境 `rl4rs`（Py3.6 + TF 1.15.0 + torch 1.9）在 4090 上 TF/PyTorch 均不可用，仅作对照。

确认文件存在：

- `output/simulator_a_dien/model.index`（DIEN 模拟器）
- `dataset/rl4rs_dataset_a_shuf.csv`
- `output/pav/Reward_SlateRecEnv_v0_a_50k_logged_pav.pt`（PAV checkpoint，Phase 1+）

## Phase 0：环境 Smoke

```bash
# 快速（约几分钟，batch=1 最省内存）
bash reproductions/run_online_phase0.sh 64 1 0 --gpu-sim

# 正式 baseline（Gate 标定）
bash reproductions/run_online_phase0.sh 512 8 0 --gpu-sim
```

输出：

- `output/qlearning_pilot/baseline_rewards.json`
- `output/qlearning_pilot/phase0_smoke_summary.md`

Gate 通过后再进 Phase 1/2。

## GPU 说明（4090 / rl4rs-tf115）

- **DIEN 模拟器（nvidia-tensorflow 1.15.5+nv23.02）**：已验证 RTX 4090 D，Compute Capability **8.9**，`tf.test.is_gpu_available() == True`。
- **PyTorch（Q 网络 + PAV）**：`torch 2.4.1+cu118`，支持 sm_89。
- **加速手段**：默认 **`--batch-size 32`**，一次并行 32 条 trajectory 过模拟器。
- **磁盘**：根分区仅 ~30G；安装/缓存请用 `export TMPDIR=/root/autodl-tmp/tmp`（数据盘 350G）。

## Phase 1–2：Q-learning Pilot

默认 **PyTorch + PAV 在 GPU**，**DIEN 模拟器 batch 并行**（TF 1.15 在本机通常仍走 CPU，但 batch=32 可显著加速）。

```bash
# raw（默认 batch=32，GPU）
python script/qlearning_train.py train --seed 0 --num-episodes 2000

# PAV
python script/qlearning_train.py train --use-pav --seed 0 --num-episodes 2000

# 强制 CPU（调试用）
python script/qlearning_train.py train --seed 0 --cpu --batch-size 1
```

或一键：

```bash
bash reproductions/run_qlearning_pilot.sh 0 32
```

输出：

- `output/qlearning_pilot/qlearning_raw_SlateRecEnv-v0_seed0.pt`
- `output/qlearning_pilot/qlearning_pav_SlateRecEnv-v0_seed0.pt`
- `output/qlearning_pilot/qlearning_*_summary.json`

**注意**：`eval` 阶段的 `sim_avg_reward` 始终在 **无 PAV 的 raw env** 上测，与 offline 主表一致。

## 模块说明

| 模块 | 作用 |
|------|------|
| `rl4rs/online/config.py` | 统一 Slate online config + Gate 阈值读取 |
| `rl4rs/online/env_utils.py` | 建 env、mask 随机动作 |
| `rl4rs/pav/online.py` | `load_pav_artifacts` + full v2 `PAVRewardWrapper` |
| `rl4rs/online/qlearning.py` | 小 MLP Q-network + TD 更新 |
| `script/online_phase0_smoke.py` | Phase 0 |
| `script/qlearning_train.py` | Phase 1–2 入口 |

## PAV Shaping 约定

### 在线 streaming（`PAVRewardWrapper`）

- 每步维护 **episode prefix**，计算 **k-step + directional** progress。
- **默认乘 Verifier**：\(C_t = p_t \cdot V_\psi(s_t,a_t)\)；归一化用 `norm_by_step`。
- 消融：`--no-verifier` → \(C_t = p_t\)，归一化用 `progress_norm_by_step`。

```bash
# 默认 PAV v2（progress × verifier）
python script/dqn_pav_pilot.py --use-pav --pav-suffix pav_v2 --seed 0 --epochs 200

# 消融：无 verifier gate
python script/dqn_pav_pilot.py --use-pav --no-verifier --pav-variant noverifier

# 消融：更小 α
python script/dqn_pav_pilot.py --use-pav --pav-alpha 0.02 --pav-variant lowalpha
```

- **eval 始终在 raw sim env** 上测。

## 内存建议

| 场景 | batch_size | 说明 |
|------|------------|------|
| 无卡 / 小内存 | 1 | Phase 0 smoke 用 64 episodes |
| GPU 正常 | 8 | Phase 0 用 512 episodes |
| Q-learning | 32（默认） | pilot 并行 simulator；PyTorch 用 GPU |
