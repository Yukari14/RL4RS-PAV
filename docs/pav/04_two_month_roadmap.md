# 两个月执行路线图

本文档将 RL4RS-only PAV 项目拆成两个月的执行路线，从精确定义 MDP 开始，到实现两网络 PAV、完成实验，再到形成 ICLR 风格论文初稿。整个计划不包含 DBRL 或 S-DPO 对接。

## 第 1 周：RL4RS MDP 与问题定义

目标是从 RL4RS 公开源码出发，把推荐过程严谨写成论文中的 MDP。

交付物：

- RL4RS MDP formulation。
- 源码实现到论文符号的对应关系。
- delayed reward credit assignment 问题定义。
- 论文 notation table 初稿。

完成标准：论文可以在不重新发明环境的前提下，准确说明 RL4RS 的状态、动作、转移、奖励和 horizon。

## 第 2 周：Reward Model、Progress 与 Verifier 目标

目标是把两网络 PAV 的方法部分全部固定下来。

交付物：

- state-only Reward Model $R_\phi(s)$ 定义。
- 非网络 progress $p_t^k$ 定义。
- progress correctness variable $Z_t$ 定义。
- Verifier $V_\psi(s,a)$ 定义。
- verified contribution $C_t$ 定义。
- reward shaping equation。

核心公式包括：

$R_\phi(s_t) \approx \mathbb{E}[G_t \mid s_t]$。

$p_t^k = \sum_{i=0}^{K_t-1}\gamma^i r_{t+i} + \mathbf{1}[K_t=k]\gamma^k R_\phi(s_{t+k}) - R_\phi(s_t)$。

$V_\psi(s_t,a_t)=P(Z_t=1\mid s_t,a_t)$。

$C_t=p_t^kV_\psi(s_t,a_t)$。

$r'_t = r_t + \alpha \cdot \text{normalize}(C_t)$。

完成标准：method section 已经有固定的 Reward Model、非网络 progress、Verifier、verified contribution 和 shaped reward 公式。

## 第 3 周：实验协议

目标是确定如何证明 idea，而不是边写代码边临时决定实验。

交付物：

- 选定环境。
- 选定 baseline。
- 选定 evaluation metrics。
- ablation plan。
- result table templates。

实验主线：$\text{CQL} \text{ vs. } \text{CQL + PAV}$，以及 $\text{BCQ} \text{ vs. } \text{BCQ + PAV}$。

完成标准：实现者不需要再决定 baseline、metric 或 ablation，可以直接按照协议跑实验。

## 第 4 周：最小 PAV 数据管线

目标是从 RL4RS 的 `MDPDataset` 中构造每个 transition 的 progress value 和 verifier label。

交付物：

- 加载 RL4RS `MDPDataset`。
- 恢复 episode transitions。
- 计算 k-step returns。
- 训练或加载 $R_\phi(s)$。
- 计算非网络 progress $p_t^k$。
- 构造 verifier label $Z_t$。

完成标准：每个 transition 都有可复现的 progress value 和 verifier label。

## 第 5 周：Verifier 与 Reward Shaping

目标是训练 verifier，并生成可以直接喂给 RL4RS offline RL 的 shaped dataset。

交付物：

- 训练 $V_\psi(s,a)$ 预测 $Z_t$。
- 保存 verifier checkpoint 和 normalization stats。
- 计算 verified contribution $C_t=p_t^kV_\psi(s_t,a_t)$。
- 生成 PAV-shaped reward。
- 导出 shaped `MDPDataset`。
- 在 `SlateRecEnv-v0` 上跑通第一次 `CQL + PAV`。

完成标准：原始 CQL 和 CQL + PAV 可以在同一套 RL4RS pipeline 下训练和评估。

## 第 6 周：主实验

目标是完成主要结果表。

交付物：

- `CQL` vs. `CQL + PAV`。
- `BCQ` vs. `BCQ + PAV`。
- 可选：`BC` vs. `BC + PAV`。
- 在 `SlateRecEnv-v0` 和 `SeqSlateRecEnv-v0` 上完成实验。

完成标准：主结果表足够判断 PAV 是否有效。

## 第 7 周：Ablation 与机制分析

目标是让论文不只是报告一个提升，而是解释为什么 PAV 有用。

交付物：

- $k$ ablation。
- $\alpha$ ablation。
- learned $R_\phi$ vs. $R_\phi=0$。
- verified contribution vs. raw progress。
- Verifier gate vs. no Verifier gate。
- progress / verifier / contribution distribution by step。
- shaped reward visualization。

完成标准：论文可以解释 PAV 为什么有效，或者某些设置为什么失败。

## 第 8 周：论文初稿

目标是完成 ICLR 风格论文初稿。

交付物：

- Introduction。
- Preliminaries: RL4RS MDP。
- Method: PAV。
- Experiments。
- Analysis。
- Limitations。

论文核心论证链：RL4RS delayed rewards 导致 action-level credit assignment 困难；Reward Model 定义状态潜力，非网络 progress 衡量局部状态改善，Verifier 判断该改善是否可靠，verified contribution reward shaping 改善 offline RL。

完成标准：初稿包含完整故事线、方法公式、实验结果、ablation、机制分析和 limitation。

