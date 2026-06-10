# PAV v2 快速 Ablation 协议

目标：在 **不跑 500 epoch 全量 RL** 的前提下，选出值得进短跑 RL pilot 的 PAV 配置。

## 三层评估（耗时递增）

### Layer 0：单元测试（秒级）

```bash
python script/test_pav_progress.py
```

覆盖 k-step progress、directional progress、necessity 标签、shape_rewards。

### Layer 1：信号级 Ablation（分钟级，主流程）

只训练 PAV 的 Reward Model + Verifier，**不训练 CQL/BCQ**：

```bash
python script/pav_ablation_fast.py \
  --env SlateRecEnv-v0 \
  --trial-name a_50k_logged \
  --max-episodes 2000 \
  --reward-epochs 3 \
  --verifier-epochs 3
```

默认 6 个变体（递进式）：

| variant | 改动 |
|---------|------|
| `baseline_sign` | 现有 sign 标签 |
| `directional` | + λ=0.5 directional progress |
| `necessity` | necessity 标签（G−R_φ proxy） |
| `directional_necessity` | 两者 + combined 标签 |
| `full_v2` | 上述 + consistency β=0.1 |
| `full_v2_no_verifier` | 无 verifier gate（对照） |

输出：`output/pav/ablation/fast_ablation_*.csv`

**Gate 条件（默认）**：

- `verifier_auc >= 0.58`（比当前 0.54 baseline 有提升）
- `contribution_return_corr >= 0.05`（shaping 信号与 return 有弱正相关）

通过 gate 的变体才进入 Layer 2。

只跑部分变体：

```bash
python script/pav_ablation_fast.py --variants baseline_sign,directional,full_v2
```

### Layer 2：短跑 RL Pilot（小时级，非 500 epoch）

对 Layer 1 中 **1–2 个** 最优变体：

1. 用对应 suffix 跑 `shape_dataset` 生成 shaped H5
2. **1 epoch** BCQ（或 DQN pilot）+ **seed=0** + 5k–10k logged subset
3. 看 action diversity + simulator raw return（现有 stability gate 流程）

```bash
# 例：仅 BCQ 1 epoch smoke
python script/pav_logged_experiment.py  # 改 trial / suffix / epoch=1
```

**不要**在 Layer 1 未通过时对 6 个变体各跑 500 epoch。

## v2 配置字段

| 字段 | 默认 | 含义 |
|------|------|------|
| `directional_lambda` | 0.0 | 0=关闭；建议 ablation 试 0.3, 0.5 |
| `embed_dim` | 64 | Reward Model 共享 embedding head |
| `verifier_label_mode` | sign | necessity / necessity_combined |
| `consistency_beta` | 0.0 | >0 启用筷子一致性 fine-tune |
| `consistency_epochs` | 2 | 一致性阶段 epoch 数 |

生成 shaped dataset 示例：

```python
from rl4rs.pav.config import PAVConfig
from rl4rs.pav.pipeline import build_pav_dataset

config = PAVConfig.from_dict({
    "env": "SlateRecEnv-v0",
    "trial_name": "a_50k_logged",
    "suffix": "pav_v2",
    "directional_lambda": 0.5,
    "verifier_label_mode": "necessity_combined",
    "consistency_beta": 0.1,
})
build_pav_dataset(config)
```

## 决策树

```text
Layer 0 pass?
  → Layer 1: 6 variants × ~3 min
    → 有 variant gate_pass?
      → Layer 2: 1–2 variants × 1 epoch BCQ
        → diversity + raw return 不降
          → 才跑 multi-seed / 更长 epoch
    → 全部 fail
      → 调 λ / margin / necessity_combined，重复 Layer 1（仍比 500 epoch 便宜）
```

## 已知限制

- **在线 streaming**（`PAVRewardWrapper`）仍用 horizon=1 potential progress，不含 directional；完整 v2 仅离线 `apply_pav_to_episode` / shaped H5。
- **真反事实 N_t** 未实现；necessity 标签为 O(1) proxy。后续可对 5% transition 用 simulator replace-rollout 作 silver label。
- **consistency** 在 verifier 弱时可能 flatten R_φ；务必看 `value_mse` 与 `progress_std` 是否 collapse。
