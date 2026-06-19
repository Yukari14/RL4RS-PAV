# RL4RS-PAV

**Process Advantage Verifiers (PAV)** — 面向 RL4RS 推荐场景的 process-level credit assignment 与 reward shaping 框架。

本项目基于 [RL4RS](https://github.com/fuxiAIlab/RL4RS)，在不改环境的前提下，用**固定参考策略 μ（Prover）** 离线估计过程优势，并在 online RL 中注入 dense shaping 信号。

[![License](https://licensebuttons.net/l/by/3.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

---

## 动机

RL4RS 中 reward 高度延迟：slate / page 完成前多步为 0，credit assignment 困难。PAV 在固定 prover μ 的坐标系下估计「这一步相对 μ 好不好」，以 advantage 形式注入 RL，而不在训练过程中重训 PAV。

---

## 核心思想（v3：冻结离线 PAV）

三组件职责：

| 组件 | 符号 | 作用 |
|------|------|------|
| **Prover** | μ | 固定参考策略（logging / BC / bo_k / random / uniform） |
| **Reward** | \(R_\phi(s) \approx V^\mu(s)\) | 状态价值；用于 k-step progress 的 bootstrap |
| **Verifier** | \(Q_\psi(s,a) \approx Q^\mu(s,a)\) | 动作价值（`q_regression` 模式） |

**k-step progress**（\(K_t=\min(k,H-t)\)）：

\[
p_t = \sum_{i=0}^{K_t-1}\gamma^i r_{t+i} + \mathbf{1}[K_t{=}k]\,\gamma^k R_\phi(s_{t+k}) - R_\phi(s_t)
\]

可选 **directional progress**（embedding 余弦差）与 \(p_t\) 组合（`directional_lambda`）。

**Contribution（v3 默认）** — advantage 而非 gate：

\[
C_t = A^\mu_t = Q_\psi(s_t,a_t) - R_\phi(s_t)
\]

**Shaped reward**：

\[
\tilde r_t = r_t + \alpha \cdot \mathrm{clip}(C_t,\,-c,\,c)
\]

（`normalize_contribution=False` 时不做 per-step z-score；可用 `use_clipping=False` 得到纯 \(\alpha A^\mu\)。）

### 工作流（两阶段，RL 阶段不重训 PAV）

```
阶段 1 — 离线 prep（每个 prover_kind + suffix 一套 artifact）
  logged MDPDataset
    → 选 prover μ，估 Q^μ（轨迹平均 + hybrid MC）
    → 训 Reward + Verifier
    → 写出 Reward_*.pt / Verifier_*.pt / stats_*.json / Policy_*.pt（BC prover）

阶段 2 — 在线 RL（DQN / PPO 等）
    → load 冻结 artifact
    → PAVRewardWrapper 每步算 C_t 并 reshape
    → eval 仍在 raw simulator 上测
```

**不做 online prover refresh / refit**：PAV 信号相对固定 μ 定义；在变化的 base policy rollout 上重训会退化。

### Prover 种类

| `prover_kind` | 说明 | 需 `Policy_*.pt` |
|---------------|------|------------------|
| `logging` | 离线日志动作 | 否 |
| `supervised` | BC 策略 | 是（可 inline 训练） |
| `bo_k` | BC + K 次采样 rerank | 是 |
| `random` / `uniform` | 规则 masked 策略 | 否 |

每种 prover 需**单独离线 prep**（不同 suffix），RL 时 `--pav-suffix` 选用。

---

## 项目结构

```
RL4RS-PAV/
├── rl4rs/
│   ├── pav/
│   │   ├── config.py          # PAVConfig
│   │   ├── prover.py          # Prover μ（logging / BC / bo_k / …）
│   │   ├── mc_estimator.py    # hybrid MC 估 Q^μ
│   │   ├── trainer.py         # build_pav_signals（离线 prep）
│   │   ├── online.py          # PAVRewardWrapper（冻结 artifact）
│   │   ├── progress.py        # k-step / directional progress
│   │   ├── suite_progress.py  # 多 prover 训练进度文件
│   │   └── training_progress.py
│   └── online/
│       ├── config.py          # default_pav_config（v3 默认）
│       └── pav_cli.py
├── script/
│   ├── pav_train_prover_suites.py  # ★ 按 prover_kind 批量训 artifact
│   ├── dqn_pav_pilot.py            # RLlib DQN raw vs PAV
│   ├── ppo_pav_pilot.py
│   ├── pav_train.py                # 离线 shape H5（CQL/BCQ 管线）
│   └── pav_ablation_fast.py
└── docs/pav/
```

---

## 环境要求

- Linux 推荐；Python 3.8+（GPU 环境见 `reproductions/setup_rl4rs_tf115_gpu.sh`）
- 数据集与 DIEN 模拟器见下方链接
- PAV hybrid MC 耗时：默认 fast 配置约 500 states × 4 rollouts × 4 CPU workers

---

## 安装

```bash
git clone https://github.com/Yukari14/RL4RS-PAV.git
cd RL4RS-PAV
export PYTHONPATH=$PYTHONPATH:$(pwd)
conda env create -f environment.yml   # 或 rl4rs-tf115 GPU 环境
conda activate rl4rs
export rl4rs_dataset_dir=$PWD/dataset
export rl4rs_output_dir=$PWD/output
```

### 数据

| 资源 | 链接 |
|------|------|
| Zenodo | https://zenodo.org/record/6622390 |
| 完整复现包 | https://drive.google.com/file/d/1YbPtPyYrMvMGOuqD4oHvK0epDtEhEb9v/view |

需要：`dataset/item_info.csv`、`dataset/rl4rs_dataset_a_shuf.csv`、`output/simulator_a_dien/model`、离线 MDPDataset（如 `SlateRecEnv-v0_a_50k_logged.h5`）。

---

## 快速开始

### 1. 训练多 prover 冻结 artifact（推荐）

```bash
# 默认：logging + supervised + bo_k；fast MC（500×4，CPU sim）
python script/pav_train_prover_suites.py --no-shaped-h5

# 指定种类 / 覆盖 MC 规模
python script/pav_train_prover_suites.py --kinds logging,supervised,bo_k \
  --max-mc-states 500 --n-mc 4 --mc-workers 4 --no-shaped-h5
```

产物（`output/pav/`）：

- `stats_*_{suffix}.json`
- `Reward_*_{suffix}.pt`
- `Verifier_*_{suffix}.pt`
- `Policy_*_{suffix}.pt`（supervised / bo_k）
- `prover_suites_manifest.json`

**进度监控**（不要用完整 TF 日志）：

```bash
watch -n 15 cat output/pav/progress_current_prover_suites.txt
tail -f output/pav/progress_live_prover_suites.txt
```

### 2. Online RL：DQN raw vs PAV

```bash
# raw baseline
python script/dqn_pav_pilot.py --epochs 150 --seed 0

# 冻结 PAV（示例：logging prover 套件）
python script/dqn_pav_pilot.py --use-pav --pav-suffix pav_v3_log \
  --pav-trial-name a_50k_logged --epochs 150 --seed 0
```

训练进度：`output/dqn_pilot/progress_live_{cond}_seed0.txt`

### 3. 离线 CQL/BCQ + shaped H5（原管线）

```bash
cd script
python pav_train.py shape_dataset "{'env':'SlateRecEnv-v0','trial_name':'a_all'}"
python batchrl_train.py CQL train "{'env':'SlateRecEnv-v0','trial_name':'a_all','use_pav':True}"
```

### 4. Python API

```python
from rl4rs.online.config import default_pav_config
from rl4rs.pav.trainer import build_pav_signals
from rl4rs.pav.dataset import load_mdpdataset

config = default_pav_config("output", "dataset", trial_name="a_50k_logged", suffix="pav_v3_log")
dataset = load_mdpdataset(config.raw_dataset_path)
signals = build_pav_signals(dataset, config)
stats = signals["stats"]
```

---

## v3 默认超参（`default_pav_config`）

| 项 | 默认 |
|----|------|
| `verifier_output_mode` | `q_regression` |
| `directional_lambda` | `0.5` |
| `verifier_label_mode` | `necessity_combined` |
| `alpha` | `0.05` |
| `max_mc_states` / `n_mc` | 500 / 4 |
| `mc_sim_use_cpu` | `True`（suite 脚本） |
| `reward_epochs` / `verifier_epochs` | 10 / 8 |

---

## 离线诊断指标（`stats_*.json`）

| 指标 | 含义 |
|------|------|
| `distinguishability` | 同 state 桶内 contribution 方差（越大越有区分度） |
| `alignment_corr` | 分 step 的 corr(\(G_t-R_\phi\), \(C_t\)) |
| `contribution_return_corr` | corr(\(C_t, G_t\)) |
| `verifier_q_mse` | \(\|Q_\psi - Q^\mu\|^2\)（主质量指标） |
| `verifier_auc` | 对 necessity 标签的 AUC（`q_regression` 下仅供参考） |

---

## 文档

| 文档 | 内容 |
|------|------|
| [`docs/pav/online_runbook.md`](docs/pav/online_runbook.md) | Online RL + GPU 环境 |
| [`docs/pav/improvement_plan.md`](docs/pav/improvement_plan.md) | Prover / Q^μ 设计笔记 |
| [`docs/pav/03_experiment_protocol.md`](docs/pav/03_experiment_protocol.md) | 实验协议 |

---

## 与上游 RL4RS

- **保留**：环境、仿真器、CQL/BCQ/BC、RLlib pilot
- **新增/更新**：`rl4rs/pav/` 冻结 v3、显式 Prover、hybrid MC、prover suite 脚本
- **移除**：online prover refresh / refit（`online_prover.py`）

上游：https://github.com/fuxiAIlab/RL4RS

---

## Citation

```bibtex
@article{2021RL4RS,
  title={RL4RS: A Real-World Benchmark for Reinforcement Learning based Recommender System},
  author={Kai Wang and Zhene Zou and Yue Shang and Qilin Deng and Minghao Zhao and Runze Wu and Xudong Shen and Tangjie Lyu and Changjie Fan},
  journal={ArXiv},
  year={2021},
  volume={abs/2110.11073}
}
```

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
