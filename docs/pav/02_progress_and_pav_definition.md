# Reward Model、Progress 与 Verifier 定义

本文档定义 RL4RS MDP 下的两网络 PAV 框架。最终版本只包含两个可训练网络：Reward Model 和 Verifier。Progress 不再作为单独网络，而是由 Reward Model 诱导出的非网络计算量。

核心分工如下：

- Reward Model 估计状态潜力，只输入状态。
- Progress 是由状态潜力差和 k-step reward 计算出的局部进步。
- Verifier 估计该局部进步是否可靠、是否与最终目标一致。
- Policy update 使用被验证过的 progress contribution。

## 1. 为什么不再使用 Progress Network

如果单独训练一个 Progress Network，它很容易和 reward model 或 Q network 发生职责重叠。为了让两个网络估计不同对象，本项目采用：

- $R_\phi(s)$：状态潜力网络，估计 state quality。
- $V_\psi(s,a)$：可靠性网络，估计 progress correctness。

Progress 本身不是网络，而是由 $R_\phi$ 在动作前后状态上的变化计算得到。因此，Progress 不直接学习 $P(Y=1 \mid s,a)$，也不变成另一个 Q function。

## 2. Reward Model：状态潜力

给定 RL4RS trajectory：$\tau = (s_0, a_0, r_0, s_1, \ldots, s_H)$。

从第 $t$ 步开始的 discounted return 为：$G_t = \sum_{i=t}^{H-1}\gamma^{i-t}r_i$。

Reward Model 只输入状态，输出状态潜力：$R_\phi(s_t) \approx \mathbb{E}[G_t \mid s_t]$。

这里 $R_\phi(s_t)$ 表示当前状态距离未来成功有多近。它不输入动作，因此不是 $Q(s,a)$。这个约束很重要，因为 PAV 要区分 state quality 和 action contribution。

Reward Model 的训练目标为：$\mathcal{L}_R = \mathbb{E}[(R_\phi(s_t)-G_t)^2]$。

如果目标是二值购买成功，也可以把 $R_\phi(s)$ 训练成成功概率：$R_\phi(s_t) \approx P(Y=1 \mid s_t)$。第一版在 RL4RS 中优先使用 expected return，因为 RL4RS reward 是 price-label 或 simulator reward，不一定是纯二值标签。

## 3. Progress：非网络局部进步

Progress 定义为 k-step temporal progress：

$p_t^k = \sum_{i=0}^{K_t-1}\gamma^i r_{t+i} + \mathbf{1}[K_t=k]\gamma^k R_\phi(s_{t+k}) - R_\phi(s_t)$。

其中 $K_t = \min(k,H-t)$。

该定义比简单的 $R_\phi(s_{t+k})-R_\phi(s_t)$ 更完整，因为它保留了 k 步内可能已经出现的 page reward 或 slate reward。对于 RL4RS 中前几步 reward 为 0 的情况，该定义会自然退化为前后状态潜力差。

直观解释：

- $R_\phi(s_t)$ 是动作前状态本来有多好。
- $\sum_{i=0}^{K_t-1}\gamma^i r_{t+i}$ 是动作后 k 步内实际得到的局部 reward。
- $R_\phi(s_{t+k})$ 是动作后 k 步状态的剩余潜力。
- $p_t^k$ 衡量动作后 trajectory 相比动作前状态是否产生了局部增益。

因此，Progress 回答的是：当前动作是否让状态在局部范围内变得更有希望？

## 4. Progress 与 Advantage 的关系

如果 $R_\phi(s)$ 逼近 base policy 的 value function $V^{\pi_{\text{base}}}(s)$，那么 $p_t^k$ 可以看作有限步 TD advantage：

$p_t^k \approx Q_k^{\pi_{\text{base}}}(s_t,a_t) - V^{\pi_{\text{base}}}(s_t)$。

这说明 Progress 的作用不是估计整个状态动作对的 future return，而是从 future return 中减去状态本身的潜力，突出动作带来的局部增量。

## 5. Verifier：Progress 可靠性

仅有 progress 不够，因为局部进步可能是虚假的。一个动作可能让短期状态潜力上升，但最终偏离用户意图，或者只是在 exploit 某个高 reward 模式。

因此定义 progress correctness variable：$Z_t \in \{0,1\}$。

$Z_t=1$ 表示：当前 transition 的局部 progress 是可靠的，也就是它不仅局部上升，而且与最终目标方向一致。

第一版可使用如下经验定义：

$Z_t = \mathbf{1}[\operatorname{sign}(p_t^k-b_p(t)) = \operatorname{sign}(G_t-\bar G(t))]$。

其中：

- $b_p(t)$ 是同 step 或同 page-position 的平均 progress baseline。
- $\bar G(t)$ 是同 step 或同 page-position 的平均 future return baseline。

这个定义表示：局部 progress 相对于同位置平均水平的方向，是否与最终 return 相对于同位置平均水平的方向一致。

如果 RL4RS 元数据中可用 item category、label 或 session target proxy，后续可加入 intent consistency：

$Z_t = \mathbf{1}[\text{outcome-consistent}(t) \land \text{intent-consistent}(t)]$。

第一版先使用 outcome consistency，intent consistency 作为增强实验或分析。

Verifier 定义为：$V_\psi(s_t,a_t)=P(Z_t=1 \mid s_t,a_t)$。

Verifier 的训练目标为：$\mathcal{L}_V = \operatorname{BCE}(V_\psi(s_t,a_t),Z_t)$。

Verifier 不是 $P(Y=1 \mid s_t,a_t)$，也不是 $Q(s,a)$。它预测的是“局部 progress 是否可靠”。

## 6. Verified Progress Contribution

定义动作贡献分数：

$C_t = p_t^k \cdot V_\psi(s_t,a_t)$。

解释：

- $p_t^k$ 给出局部进步的方向和幅度。
- $V_\psi(s_t,a_t)$ 给出该局部进步可靠的概率。
- $C_t$ 表示经过 verifier 校准后的 process contribution。

四种情况：

- $p_t^k>0$ 且 $V_\psi$ 高：可靠的正贡献。
- $p_t^k>0$ 且 $V_\psi$ 低：看似进步但不可靠，贡献被压低。
- $p_t^k<0$ 且 $V_\psi$ 高：可靠的负贡献。
- $p_t^k<0$ 且 $V_\psi$ 低：负判断不可靠，负贡献被压低。

这个乘积不是两个 reward model 的叠加，而是“状态潜力诱导的局部增益”与“局部增益可靠性”的组合。

## 7. Reward Shaping

PAV 通过 shaped reward 接入 RL4RS offline RL：

$r'_t = r_t + \alpha \cdot \operatorname{clip}(\operatorname{norm}(C_t),-c,c)$。

默认设置为 $\alpha=0.1$，$c=3$。

其中 normalization 必须在环境内完成，避免 $C_t$ 的尺度破坏 CQL 或 BCQ 的 reward scale。

最终 shaped dataset 为：$\text{MDPDataset}(\text{observations}, \text{actions}, \text{shaped\_rewards}, \text{terminals})$。

## 8. 关于 Disagreement 的位置

不建议把 disagreement penalty 作为第一版主方法。原因是 $p_t^k$ 是实数，$V_\psi(s_t,a_t)$ 是概率，两者不能直接相减。

如果需要分析两者冲突，必须先定义：$q_t=\sigma(p_t^k/\tau)$。

然后 disagreement 可以写成：$D_t=|q_t - V_\psi(s_t,a_t)|$。

但 $D_t$ 更适合作为：

- active sampling 指标；
- uncertainty analysis；
- early exploration bonus；
- ablation。

第一版主方法保持为 verified contribution：$C_t=p_t^kV_\psi(s_t,a_t)$。

## 9. 核心主张

两网络 PAV 的核心不是训练两个 reward model，而是明确拆分：

- Reward Model：当前状态有多好。
- Progress：动作后状态相对动作前状态改善多少。
- Verifier：这个改善是否可靠、是否与最终目标一致。

因此，PAV 回答的是 action-level credit assignment 问题：在 delayed reward 的 RL4RS 中，哪个 action 对最终成功产生了被验证过的过程贡献？

