# RL4RS-PAV

**Process Advantage Verifiers (PAV)** — 面向 RL4RS 离线推荐场景的 process-level credit assignment 与 reward shaping 框架。

本项目基于 [RL4RS](https://github.com/fuxiAIlab/RL4RS) 公开代码与 MDP 建模，在不修改环境的前提下，为 offline RL 提供**被验证过的过程贡献信号**，缓解 delayed reward 带来的 credit assignment 困难。

[![License](https://licensebuttons.net/l/by/3.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

---

## 动机：RL4RS 的 credit assignment 问题

在 RL4RS 中，单个 item 推荐动作的影响往往要等到 **slate 完成**或 **page 结束**才体现在 reward 上。原始离线数据里，前面多步 reward 为 0，最终一步才出现聚合收益。

这带来两个问题：

1. **状态质量 vs 动作贡献混淆**：一个本来就很「有希望」的状态，会让任意动作看起来都不错。
2. **虚假局部进步**：某些动作可能短期提升状态潜力，但最终偏离用户真实意图。

PAV 的目标是把这两个问题拆开，并只把**可靠的动作增量**注入 offline RL 训练。

---

## 核心思想

PAV 使用**两个网络 + 一个非网络量**，职责明确分离：

| 组件 | 符号 | 输入 | 估计对象 |
|------|------|------|----------|
| Reward Model | $R_\phi(s)$ | 状态 $s$ | 状态潜力 $\mathbb{E}[G_t \mid s_t]$ |
| Progress（非网络） | $p_t^k$ | 由 $R_\phi$ 与轨迹计算 | 动作的 k 步局部进步 |
| Verifier | $V_\psi(s,a)$ | 状态 + 动作 | 局部进步是否可靠 $P(Z_t{=}1 \mid s,a)$ |

**关键约束**：$R_\phi$ 不输入动作，因此不是 $Q(s,a)$；$V_\psi$ 预测的是 progress 可靠性，也不是最终成功概率。

### 方法流程

```
离线 MDPDataset
    │
    ├─► 训练 R_φ(s)     目标：MSE(R_φ, G_t)
    │
    ├─► 计算 p_t^k      非网络：k-step reward + 状态潜力差
    │
    ├─► 生成标签 Z_t    outcome consistency（同 step baseline 比较）
    │
    ├─► 训练 V_ψ(s,a)   目标：BCE(V_ψ, Z_t)
    │
    ├─► C_t = p_t^k × V_ψ(s,a)    verified contribution
    │
    └─► r'_t = r_t + α · clip(norm(C_t))    shaped reward
            → 导出新的 MDPDataset → CQL / BCQ / BC 训练
```

### 核心公式

**k-step Progress**（$K_t = \min(k, H-t)$）：

$$p_t^k = \sum_{i=0}^{K_t-1}\gamma^i r_{t+i} + \mathbf{1}[K_t{=}k]\,\gamma^k R_\phi(s_{t+k}) - R_\phi(s_t)$$

**Verifier 标签**（第一版 outcome consistency）：

$$Z_t = \mathbf{1}\big[\operatorname{sign}(p_t^k - b_p(t)) = \operatorname{sign}(G_t - \bar G(t))\big]$$

**Verified Contribution & Shaped Reward**：

$$C_t = p_t^k \cdot V_\psi(s_t, a_t), \qquad r'_t = r_t + \alpha \cdot \operatorname{clip}(\operatorname{norm}(C_t),\,-c,\,c)$$

默认超参：$\alpha{=}0.1$，$c{=}3$，$k{=}3$（`SlateRecEnv-v0`）/ $k{=}5$（`SeqSlateRecEnv-v0`）。

更完整的理论定义见 [`docs/pav/`](docs/pav/)。

---

## 项目结构

```
RL4RS-main/
├── rl4rs/
│   ├── env/              # RL4RS 环境（SlateRecEnv / SeqSlateRecEnv）
│   ├── nets/             # 仿真器与 offline RL 网络
│   └── pav/              # ★ PAV 核心实现
│       ├── config.py     # 超参与路径配置
│       ├── dataset.py    # MDPDataset 展开与导出
│       ├── models.py     # RewardModel / Verifier
│       ├── progress.py   # p_t^k, Z_t, reward shaping
│       ├── trainer.py    # 训练与信号编排
│       └── pipeline.py   # 对外入口 build_pav_dataset()
├── script/
│   ├── pav_train.py      # PAV 命令行入口
│   └── batchrl_train.py  # offline RL 训练（支持 use_pav）
├── reproductions/
│   └── run_pav.sh        # 端到端复现脚本
└── docs/pav/             # MDP 建模、PAV 定义、实验协议
```

**PAV 接入点**：在 `MDPDataset` 生成之后、offline RL 训练之前，不侵入 `rl4rs/env/`。

---

## 环境要求

- Linux（推荐）或 Windows
- Python 3.6+，Conda
- 至少 64 GB 内存（RL4RS 原始要求）
- GPU（训练 Reward Model / Verifier / CQL 时推荐）

---

## 安装

```bash
git clone https://github.com/Yukari14/RL4RS-PAV.git
cd RL4RS-PAV
export PYTHONPATH=$PYTHONPATH:$(pwd)
conda env create -f environment.yml
conda activate rl4rs
```

### 数据集下载

本仓库**不包含**大型数据文件（`dataset/*.csv`、`dataset/*.h5` 已在 `.gitignore` 中排除）。请从 RL4RS 官方渠道下载后放到 `dataset/` 目录：

| 资源 | 链接 |
|------|------|
| 数据（仅数据） | https://zenodo.org/record/6622390 |
| 完整复现包 | https://drive.google.com/file/d/1YbPtPyYrMvMGOuqD4oHvK0epDtEhEb9v/view |

至少需要：

- `dataset/item_info.csv`（已随仓库提供）
- `dataset/rl4rs_dataset_a_shuf.csv`（`SlateRecEnv-v0`）
- `dataset/rl4rs_dataset_b3_shuf.csv`（`SeqSlateRecEnv-v0`）
- `output/simulator_a_dien/model`（仿真器，需先训练或从复现包获取）

---

## 快速开始

### 1. 仅运行 PAV reward shaping

```bash
export rl4rs_dataset_dir=../dataset
export rl4rs_output_dir=../output

cd script
python pav_train.py shape_dataset "{'env':'SlateRecEnv-v0','trial_name':'a_all'}"
```

输出：

- `dataset/SlateRecEnv-v0_a_all_pav.h5` — shaped 离线数据集
- `output/pav/Reward_*.pt` — Reward Model 权重
- `output/pav/Verifier_*.pt` — Verifier 权重
- `output/pav/stats_*.json` — 训练统计

### 2. 生成诊断报告

```bash
python pav_train.py diagnostics "{'env':'SlateRecEnv-v0','trial_name':'a_all'}"
```

### 3. 端到端：PAV + Offline RL

```bash
cd reproductions
bash run_pav.sh CQL SlateRecEnv-v0 a_all
```

流程：`dataset_generate` → `PAV shape` → `PAV diagnostics` → `CQL train` → `eval` → `OPE`

训练时通过 `use_pav=True` 自动加载 shaped 数据集：

```python
python batchrl_train.py CQL train "{'env':'SlateRecEnv-v0','trial_name':'a_all','use_pav':True}"
```

### 4. Python API

```python
from rl4rs.pav import PAVConfig, build_pav_dataset

config = PAVConfig.from_dict({
    "env": "SlateRecEnv-v0",
    "trial_name": "a_all",
    "alpha": 0.1,
    "k": 3,
})
shaped_path, stats = build_pav_dataset(config)
```

---

## 实验对比

主实验对比 **Offline RL（原始 reward）** vs **Offline RL + PAV（shaped reward）**：

| 算法 | 原始 | + PAV |
|------|:----:|:-----:|
| CQL | ✓ | ✓ |
| BCQ | ✓ | ✓ |
| BC | ✓ | ✓ |

支持环境：

- `SlateRecEnv-v0`（9-step slate，$H=9$）
- `SeqSlateRecEnv-v0`（4-page sequential slate，$H=36$）

详细协议见 [`docs/pav/03_experiment_protocol.md`](docs/pav/03_experiment_protocol.md)。

---

## 消融实验

通过 `PAVConfig` 或 `pav_train.py` 的 `extra_config` 控制：

| 配置项 | 默认值 | 消融用途 |
|--------|--------|----------|
| `use_verifier` | `True` | 关闭 Verifier，$C_t = p_t^k$ |
| `use_raw_progress` | `False` | 直接用 progress，不乘 Verifier 分数 |
| `reward_model_zero` | `False` | $R_\phi \equiv 0$，测试无状态潜力时的 progress |
| `alpha` | `0.1` | shaping 强度（如 0.05 / 0.2） |
| `k` | 3 或 5 | progress 视野长度 |

```bash
python pav_train.py shape_dataset "{'env':'SlateRecEnv-v0','use_verifier':False,'suffix':'pav_noverifier'}"
python batchrl_train.py CQL train "{'env':'SlateRecEnv-v0','use_pav':True,'pav_suffix':'pav_noverifier'}"
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [`01_rl4rs_mdp_formulation.md`](docs/pav/01_rl4rs_mdp_formulation.md) | RL4RS MDP 建模与符号对应 |
| [`02_progress_and_pav_definition.md`](docs/pav/02_progress_and_pav_definition.md) | PAV 完整数学定义 |
| [`03_experiment_protocol.md`](docs/pav/03_experiment_protocol.md) | 主实验与消融协议 |
| [`04_two_month_roadmap.md`](docs/pav/04_two_month_roadmap.md) | 研究路线图 |

---

## 与上游 RL4RS 的关系

本项目是 RL4RS 的**扩展 fork**，保留其全部环境与 baseline 能力：

- **保留**：`SlateRecEnv-v0`、`SeqSlateRecEnv-v0`、仿真器训练、CQL/BCQ/BC 等 offline RL pipeline
- **新增**：`rl4rs/pav/` 子包，在 MDPDataset 层做 verified progress shaping
- **不修改**：环境 transition、原始 reward 计算、数据生成逻辑

上游资源：

- RL4RS 论文：https://arxiv.org/pdf/2110.11073.pdf
- 原始仓库：https://github.com/fuxiAIlab/RL4RS
- Tutorial：https://github.com/fuxiAIlab/RL4RS/blob/main/tutorial.ipynb

---

## Citation

如果使用本仓库，请引用 RL4RS 原文：

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

本项目继承 RL4RS 的 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。
