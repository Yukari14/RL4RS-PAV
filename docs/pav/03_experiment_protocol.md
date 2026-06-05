# RL4RS 上的 PAV 实验协议

本文档固定 Process Advantage Verifiers（PAV）在 RL4RS 上的第一版实验协议。实验目标是验证：在 delayed reward 场景中，由 Reward Model 诱导出的 progress 和 Verifier 校准后的 verified contribution 是否能提升 RL4RS offline RL 的表现。

## 1. 核心问题

主问题是：process-level advantage signal 是否能比原始 delayed reward 更好地训练 RL4RS offline RL policy？

主对比为：$\text{Offline RL(original reward)} \text{ vs. } \text{Offline RL(verified-progress-shaped reward)}$。

## 2. 环境

只使用 RL4RS 公开环境：

- `SlateRecEnv-v0`
- `SeqSlateRecEnv-v0`

第一版使用 discrete action setting：$\texttt{support\_conti\_env} = \text{False}$。

默认 horizon 为：$H = 9$ for `SlateRecEnv-v0`，$H = 36$ for `SeqSlateRecEnv-v0`。

## 3. 算法

主实验算法：

- `CQL`
- `BCQ`

辅助算法：

- `BC`

不纳入 DBRL 或 S-DPO。它们不是本项目当前范围。

## 4. 方法变体

对每个 offline RL 算法，对比：$\text{Algo} \text{ vs. } \text{Algo + PAV}$。

主表包含：

- `BCQ`
- `BCQ + PAV`
- `CQL`
- `CQL + PAV`
- `BC`
- `BC + PAV`

其中 `BC` 用来观察 PAV 是否在非 Q-learning 设置下也能提供帮助。论文主结论优先基于 `BCQ` 和 `CQL`。

## 5. PAV 默认设置

默认方法配置：

- Reward Model：state-only value predictor $R_\phi(s)$。
- Progress：非网络，由 k-step reward 和 $R_\phi$ 的前后状态潜力差计算。
- Verifier：binary reliability verifier $V_\psi(s,a)$。
- target：progress correctness variable $Z_t$。
- reward shaping：additive shaping。

默认 $k$ 值：$k = 3$ for `SlateRecEnv-v0`，$k = 5$ for `SeqSlateRecEnv-v0`。

默认 progress：$p_t^k = \sum_{i=0}^{K_t-1}\gamma^i r_{t+i} + \mathbf{1}[K_t=k]\gamma^k R_\phi(s_{t+k}) - R_\phi(s_t)$。

默认 verified contribution：$C_t = p_t^k V_\psi(s_t,a_t)$。

默认 shaped reward：$r'_t = r_t + \alpha \cdot \text{clip}(\text{normalize}(C_t), -3, 3)$。

默认 $\alpha = 0.1$。

## 6. 评估指标

使用 RL4RS 现有 evaluation path。

Simulator evaluation：

- average episode reward
- reward mean / std

Offline policy evaluation：

- `CIPS`
- `DR`
- `WIPS`
- `SeqDR`

如果计算资源允许，报告多 seed 的 mean 和 standard deviation。

## 7. Ablation

必须完成的 ablation：

- $k \in \{1, 3, 5\}$
- $\alpha \in \{0.05, 0.1, 0.2\}$
- learned $R_\phi$ vs. $R_\phi = 0$
- verified contribution vs. raw progress
- Verifier gate vs. no Verifier gate
- with clipping vs. without clipping

Reward Model / Verifier capacity ablation：

- small MLP
- medium MLP
- large MLP

预期结论不是网络越大越好，而是观察 moderate Reward Model 和 Verifier 是否给出更稳定的 process signal。

## 8. 表格模板

主结果表：

| Environment | Algorithm | PAV | Avg Reward | CIPS | DR | WIPS | SeqDR |
|---|---|---:|---:|---:|---:|---:|---:|
| SlateRecEnv-v0 | CQL | No |  |  |  |  |  |
| SlateRecEnv-v0 | CQL | Yes |  |  |  |  |  |
| SlateRecEnv-v0 | BCQ | No |  |  |  |  |  |
| SlateRecEnv-v0 | BCQ | Yes |  |  |  |  |  |

Ablation 表：

| Environment | Algorithm | $k$ | $\alpha$ | $R_\phi$ | Shaping Signal | Avg Reward | SeqDR |
|---|---|---:|---:|---|---|---:|---:|
| SlateRecEnv-v0 | CQL + PAV | 1 | 0.1 | learned | $C_t$ |  |  |
| SlateRecEnv-v0 | CQL + PAV | 3 | 0.1 | learned | $C_t$ |  |  |
| SlateRecEnv-v0 | CQL + PAV | 5 | 0.1 | learned | $C_t$ |  |  |

Reward Model / Verifier 表：

| Environment | Reward Model Size | Verifier Size | Value MSE | Verifier AUC | Avg Reward | SeqDR |
|---|---|---:|---:|---:|
| SlateRecEnv-v0 | small | small |  |  |  |  |
| SlateRecEnv-v0 | medium | medium |  |  |  |  |
| SlateRecEnv-v0 | large | large |  |  |  |  |

## 9. 诊断图

需要生成的图：

- step index vs. average original reward
- step index vs. average progress $p_t^k$
- step index vs. average verifier score $V_\psi(s_t,a_t)$
- step index vs. average contribution $C_t$
- step index vs. shaped reward
- $k$ vs. performance
- $\alpha$ vs. performance
- $Z_t$ label distribution
- optional disagreement $D_t$ distribution

这些图要展示 verified progress 是否能在 delayed slate/page reward 到来前提供非零且可靠的学习信号。

## 10. 验收标准

最低成功标准：$\text{CQL + PAV} \geq \text{CQL}$ on `SlateRecEnv-v0` average reward。

强成功标准：$\text{CQL + PAV} > \text{CQL}$，且 $\text{BCQ + PAV} > \text{BCQ}$，并且上述提升在 `SlateRecEnv-v0` 和 `SeqSlateRecEnv-v0` 上都能观察到。

机制成功标准：progress 在 terminal/page reward 到来之前已经具有信息量，Verifier 能过滤虚假 progress，并且 $C_t$ 优于 raw progress。

如果主 reward metric 没有提升，也必须分析失败原因来自 target construction、reward scaling、verifier quality，还是 offline RL 本身的不稳定性。
