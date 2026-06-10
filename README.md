# RL4RS-PAV

**Process Advantage Verifiers (PAV)** — 面向 RL4RS 推荐场景的 process-level credit assignment 与 reward shaping 框架。

本项目基于 [RL4RS](https://github.com/fuxiAIlab/RL4RS) 公开代码与 MDP 建模，在不修改环境 transition 的前提下，为 **离线 RL**（CQL / BCQ / BC）与 **在线 RL**（Q-learning / RLlib DQN·PPO）提供**被验证过的过程贡献信号**，缓解 delayed reward 带来的 credit assignment 困难。

[![License](https://licensebuttons.net/l/by/3.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

---

## 动机

在 RL4RS 中，单个 item 推荐动作的影响往往要等到 **slate 完成**或 **page 结束**才体现在 reward 上。原始数据里前几步 reward 常为 0，最后一步才出现聚合收益。

这带来两个问题：

1. **状态质量 vs 动作贡献混淆**：高潜力状态下，任意动作看起来都不错。
2. **虚假局部进步**：短期状态潜力上升，但最终偏离用户意图。

PAV 用 **Reward Model（状态潜力）+ Progress（动作增量）+ Verifier（增量可靠性）** 拆开上述问题，只把可靠的过程贡献注入 RL 训练。

---

## 核心思想

PAV 使用**两个网络 + 一个非网络量**，职责明确分离：

| 组件 | 符号 | 输入 | 估计对象 |
|------|------|------|----------|
| Reward Model | $R_\phi(s)$ | 状态 $s$ | 状态潜力 $\mathbb{E}[G_t \mid s_t]$；可选 embed $e(s)$ |
| Progress（非网络） | $p_t$ | 由 $R_\phi$ 与轨迹计算 | k 步局部进步 + 可选方向项 |
| Verifier | $V_\psi(s,a)$ | 状态 + 动作 | 局部进步可靠性 $P(Z_t{=}1 \mid s,a)$ |

**关键约束**：

- $R_\phi$ **不输入动作**，因此不是 $Q(s,a)$。
- $V_\psi$ 预测 **progress 可靠性**，不是最终购买/成功概率。
- Progress **不是第三个网络**，而是由 $R_\phi$ 在轨迹上诱导出的标量。

### 方法流程

```
MDPDataset 或在线 rollout
    │
    ├─► 训练 R_φ(s) [+ embed head]     目标：MSE(R_φ, G_t)
    │
    ├─► 计算 p_t^k                     k-step reward + bootstrap R_φ(s_{t+k}) − R_φ(s_t)
    │
    ├─► [可选] 方向项                  p_t = p_t^k + λ·(cos(e_{t+K}, g) − cos(e_t, g))，g = e(s_0)
    │
    ├─► 生成标签 Z_t                   outcome sign / necessity / 组合（同 step baseline）
    │
    ├─► 训练 V_ψ(s,a)                  目标：BCE(V_ψ, Z_t)
    │
    ├─► [可选] Consistency 微调 R_φ    低 V_ψ 时惩罚 |progress|，减轻虚假局部信号
    │
    ├─► C_t = p_t × V_ψ(s,a)           verified contribution（Verifier 默认开启）
    │
    └─► r'_t = r_t + α · clip(norm(C_t))
            │
            ├─► 离线：导出 shaped H5 → CQL / BCQ / BC
            └─► 在线：PAVRewardWrapper 训练；eval 仍用 raw simulator reward
```

### 核心公式

符号：$t$ 为当前步，$K_t = \min(k, H-t)$，$G_t$ 为从 $t$ 起的回报，$R_\phi(s)$ 为状态潜力，$V_\psi(s,a)$ 为 Verifier 分数。

#### 1. k-step Progress

$$
p_t^k = \sum_{i=0}^{K_t-1} \gamma^i r_{t+i} + \mathbf{1}[K_t{=}k]\,\gamma^k R_\phi(s_{t+k}) - R_\phi(s_t)
$$

| 项 | 含义 |
|----|------|
| $\sum_{i=0}^{K_t-1} \gamma^i r_{t+i}$ | 动作后 $K_t$ 步内**已经到账**的 reward |
| $\gamma^k R_\phi(s_{t+k})$ | 仅当 $K_t{=}k$ 时 bootstrap 剩余潜力 |
| $- R_\phi(s_t)$ | 减去动作前状态潜力，只保留**动作增量** |

#### 2. Directional Progress（`directional_lambda > 0`）

Reward Model 共享 trunk，另接 L2 归一化 **embed head** $e(s)$：

$$
g = e(s_0), \qquad
\Delta_t^{\text{dir}} = \cos\!\big(e(s_{t+K}),\, g\big) - \cos\!\big(e(s_t),\, g\big), \qquad
p_t = p_t^k + \lambda\,\Delta_t^{\text{dir}}
$$

| 配置 | 默认 | 推荐 |
|------|------|------|
| `directional_lambda` ($\lambda$) | 0 | 0.5 |
| `embed_dim` | 64 | 64 |

#### 3. Verifier 标签（`verifier_label_mode`）

| 模式 | 标签定义 |
|------|----------|
| `sign` | $Z_t{=}1 \iff \text{sign}(p_t - b_p(t)) = \text{sign}(G_t - \bar{G}(t))$ |
| `necessity` | $Z_t{=}1 \iff (G_t - R_\phi(s_t)) > b_N(t)$ |
| `necessity_combined`（**推荐**） | 上式 necessity **且** $(p_t - b_p(t)) > m_p(t)$ |

$b_p(t)$、$\bar{G}(t)$、$b_N(t)$ 为同 step 上的 batch 均值；excess 需超过 `verifier_margin_frac` $\times$ 标准差（默认 0.25）。

#### 4. Consistency 微调 $R_\phi$（可选，Verifier 训练后）

Verifier 训完后，对 $R_\phi$ 做短阶段微调，**压低「Verifier 认为不可靠、但 $R_\phi$ 仍给出大 progress」的 transition**：

$$
\mathcal{L} = \mathcal{L}_R + \beta\,\mathbb{E}\Big[\big(1 - V_\psi(s_t,a_t)\big)\,\big|p_t\big|\Big]
$$

其中 $\mathcal{L}_R = \mathbb{E}\big[(R_\phi(s_t) - G_t)^2\big]$，$\beta$ 为 `consistency_beta`（$>0$ 启用，推荐 0.1），$|p_t|$ 为**可微** k-step progress 幅度。

开启 directional 时，微调阶段

$$
p_t = p_t^k + \lambda\,\Delta_t^{\text{dir}}
$$

方向项 $\Delta_t^{\text{dir}}$ **当常数**（不对 embed 反传）。

| 配置 | 默认 | 推荐 |
|------|------|------|
| `consistency_beta` | 0 | 0.1 |
| `consistency_epochs` | 2 | 2 |

实现：`rl4rs/pav/trainer.py` → `finetune_reward_consistency()`。

#### 5. Verified Contribution & Shaped Reward

$$
C_t = p_t \cdot V_\psi(s_t, a_t), \qquad
r'_t = r_t + \alpha \cdot \text{clip}\!\big(\text{norm}(C_t),\,-c,\,c\big)
$$

| $p_t$ | $V_\psi$ 高 | $V_\psi$ 低 |
|-------|-------------|-------------|
| $> 0$ | 可靠正贡献 | 看似进步，贡献被压低 |
| $< 0$ | 可靠负贡献 | 负判断不可靠，被压低 |

| 超参 | 值 |
|------|-----|
| $\alpha$ | 0.1 |
| $c$ | 3 |
| $k$ | 3（SlateRecEnv）/ 5（SeqSlateRecEnv） |

更完整的理论定义见 [`docs/pav/`](docs/pav/)。

---

## 推荐配置

离线 shaping 与在线 pilot 的常用默认值（`rl4rs/online/config.py` → `default_pav_config`）：

```python
RECOMMENDED = {
    "suffix": "pav_v2",
    "directional_lambda": 0.5,
    "embed_dim": 64,
    "verifier_label_mode": "necessity_combined",
    "consistency_beta": 0.1,
    "use_verifier": True,
    "alpha": 0.1,
    "clip_c": 3.0,
}
```

---

## 两条使用路径

| 路径 | Progress 计算 | 训练 reward | 评估 reward |
|------|---------------|-------------|-------------|
| **离线** | 完整 k-step + directional + consistency | shaped `.h5` | simulator raw |
| **在线** | `PAVRewardWrapper` 维护 episode 前缀，逐步算完整 p_t | shaped | **raw**（wrapper `enabled=False`） |

在线 wrapper（`rl4rs/pav/online.py`）在 vector env 中为每条 trajectory 缓存前缀，每步重算 k-step + directional progress 并乘 Verifier，与离线定义一致。

---

## 项目结构

```
RL4RS-PAV/
├── rl4rs/
│   ├── env/                 # SlateRecEnv / SeqSlateRecEnv
│   ├── nets/                # 仿真器、CQL/BCQ、RLlib 模型
│   ├── pav/
│   │   ├── config.py        # PAVConfig（directional / necessity / consistency 等）
│   │   ├── models.py        # RewardModel + embed head, Verifier
│   │   ├── progress.py      # k-step / directional progress, verifier labels
│   │   ├── trainer.py       # fit R_φ / V_ψ, consistency fine-tune
│   │   ├── online.py        # frozen inference, PAVRewardWrapper
│   │   └── pipeline.py      # build_pav_dataset()
│   └── online/
│       ├── qlearning.py     # PyTorch masked Q-learning
│       ├── config.py        # env / default_pav_config
│       └── pav_cli.py       # --pav-suffix, --no-verifier 等
├── script/
│   ├── pav_train.py         # 离线 fit + shape H5
│   ├── batchrl_train.py     # CQL / BCQ / BC（use_pav）
│   ├── dqn_pav_pilot.py     # RLlib DQN × raw / PAV
│   ├── ppo_pav_pilot.py     # RLlib PPO × raw / PAV
│   ├── qlearning_train.py   # PyTorch Q-learning pilot
│   ├── pav_ablation_fast.py # 快速消融
│   └── online_phase0_smoke.py
├── reproductions/           # run_pav.sh, run_qlearning_pilot.sh, run_dqn_pav_pilot.sh 等
└── docs/pav/                # 理论、实验协议、online_runbook
```

PAV 不修改 `rl4rs/env/` 的 transition 与原始 reward 定义。

---

## 环境要求

- Linux（推荐）；Python 3.6+（在线 pilot 推荐 3.8 + `nvidia-tensorflow`）；Conda；≥64 GB RAM（完整 offline 流程）
- GPU：训练 R_φ / V_ψ、在线 DQN/PPO 推荐
- RTX 4090：仿真器需 `bash reproductions/setup_rl4rs_tf115_gpu.sh` 安装 TF 1.15.5+nv23.02；详见 [`docs/pav/online_runbook.md`](docs/pav/online_runbook.md)

---

## 安装

```bash
git clone https://github.com/Yukari14/RL4RS-PAV.git
cd RL4RS-PAV
export PYTHONPATH=$PYTHONPATH:$(pwd)
conda env create -f environment.yml
conda activate rl4rs
```

### 数据集

大型文件不在仓库内。从 RL4RS 官方下载后放入 `dataset/`：

| 资源 | 链接 |
|------|------|
| 数据 | https://zenodo.org/record/6622390 |
| 完整复现包 | https://drive.google.com/file/d/1YbPtPyYrMvMGOuqD4oHvK0epDtEhEb9v/view |

至少需要 `item_info.csv`、`rl4rs_dataset_a_shuf.csv`、`rl4rs_dataset_b3_shuf.csv`，以及 `output/simulator_a_dien/model`。

---

## 快速开始

### 1. 离线：PAV reward shaping

```bash
export rl4rs_dataset_dir=../dataset
export rl4rs_output_dir=../output

cd script
python pav_train.py shape_dataset "{'env':'SlateRecEnv-v0','trial_name':'a_all','suffix':'pav_v2','directional_lambda':0.5,'verifier_label_mode':'necessity_combined','consistency_beta':0.1}"
```

输出：`dataset/*_pav_v2.h5`、`output/pav/Reward_*.pt`、`Verifier_*.pt`、`stats_*.json`。

### 2. 诊断

```bash
python pav_train.py diagnostics "{'env':'SlateRecEnv-v0','trial_name':'a_all','suffix':'pav_v2'}"
```

### 3. 离线 RL

```bash
cd reproductions
bash run_pav.sh CQL SlateRecEnv-v0 a_all
```

或指定 shaped 后缀：

```bash
python batchrl_train.py CQL train "{'env':'SlateRecEnv-v0','trial_name':'a_all','use_pav':True,'pav_suffix':'pav_v2'}"
```

### 4. 在线：DQN / PPO pilot

训练阶段 `--use-pav` 启用 `PAVRewardWrapper`；**评估**在 raw DIEN simulator 上测 return。

```bash
export rl4rs_output_dir=$PWD/output
export rl4rs_dataset_dir=$PWD/dataset

# DQN raw vs PAV
python script/dqn_pav_pilot.py --seed 0 --epochs 100
python script/dqn_pav_pilot.py --use-pav --pav-suffix pav_v2 --seed 0 --epochs 100

# PPO raw vs PAV
python script/ppo_pav_pilot.py --seed 0 --epochs 100
python script/ppo_pav_pilot.py --use-pav --pav-suffix pav_v2 --seed 0 --epochs 100
```

### 5. 在线：PyTorch Q-learning

```bash
python script/qlearning_train.py train --seed 0 --num-episodes 2000
python script/qlearning_train.py train --use-pav --seed 0 --num-episodes 2000
```

一键脚本：`bash reproductions/run_qlearning_pilot.sh 0 32`

### 6. Python API

```python
from rl4rs.pav import PAVConfig, build_pav_dataset
from rl4rs.pav.online import load_pav_artifacts, PAVRewardWrapper
from rl4rs.online.config import default_pav_config
from rl4rs.online.env_utils import make_slate_env

# 离线
cfg = PAVConfig.from_dict(default_pav_config("../output", "../dataset").__dict__)
shaped_path, stats = build_pav_dataset(cfg)

# 在线
artifacts = load_pav_artifacts(cfg)
env = PAVRewardWrapper(make_slate_env(env_cfg), artifacts=artifacts, enabled=True)
```

完整在线流程见 [`docs/pav/online_runbook.md`](docs/pav/online_runbook.md)。

---

## 实验

| 设置 | 离线 | 在线 |
|------|------|------|
| 算法 | CQL / BCQ / BC | Q-learning / RLlib DQN·PPO |
| 对比 | raw reward vs shaped H5 | raw env vs PAVRewardWrapper |
| 环境 | SlateRecEnv (H=9), SeqSlateRecEnv (H=36) | 同上 |

协议：[`03_experiment_protocol.md`](docs/pav/03_experiment_protocol.md)、[`06_v2_ablation_protocol.md`](docs/pav/06_v2_ablation_protocol.md)。

---

## 消融

通过 `PAVConfig` 或 `pav_train.py` / `pav_cli.py` 控制：

| 配置项 | 默认 | 用途 |
|--------|------|------|
| `directional_lambda` | 0（推荐 0.5） | 0 关闭方向项 |
| `verifier_label_mode` | `sign`（推荐 `necessity_combined`） | Verifier 标签定义 |
| `consistency_beta` | 0（推荐 0.1） | 0 跳过 consistency 微调 |
| `use_verifier` | `True` | 关闭后 C_t = p_t |
| `use_raw_progress` | `False` | 不乘 V_ψ 分数 |
| `reward_model_zero` | `False` | R_φ ≡ 0 |
| `alpha` / `k` | 0.1 / 3 或 5 | shaping 强度与视野 |
| `suffix` | `pav` | artifact 与 H5 后缀 |

快速消融：`python script/pav_ablation_fast.py`

```bash
python pav_train.py shape_dataset "{'env':'SlateRecEnv-v0','use_verifier':False,'suffix':'pav_noverifier'}"
python batchrl_train.py CQL train "{'env':'SlateRecEnv-v0','use_pav':True,'pav_suffix':'pav_noverifier'}"
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [`01_rl4rs_mdp_formulation.md`](docs/pav/01_rl4rs_mdp_formulation.md) | MDP 建模 |
| [`02_progress_and_pav_definition.md`](docs/pav/02_progress_and_pav_definition.md) | 完整数学定义 |
| [`03_experiment_protocol.md`](docs/pav/03_experiment_protocol.md) | 离线实验协议 |
| [`online_runbook.md`](docs/pav/online_runbook.md) | 在线 RL + GPU 环境 |
| [`06_v2_ablation_protocol.md`](docs/pav/06_v2_ablation_protocol.md) | 消融矩阵 |

---

## 与 RL4RS 的关系

Fork 自 [RL4RS](https://github.com/fuxiAIlab/RL4RS)：**保留**全部环境与 offline baseline；**新增** `rl4rs/pav/`、`rl4rs/online/`；**不修改**环境 transition 与 logged 数据生成。

- 论文：https://arxiv.org/pdf/2110.11073.pdf
- 原仓库：https://github.com/fuxiAIlab/RL4RS

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

PAV 方法论文投稿中，引用信息待更新。

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
