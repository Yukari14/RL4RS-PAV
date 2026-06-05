# 面向 Process Advantage Verifiers 的 RL4RS MDP 建模

本文档固定 PAV 的研究起点：直接使用 RL4RS 公开源码中的 MDP 建模，不重新定义新的推荐环境。这里要把源码中的状态、动作、转移、奖励、终止条件和离线数据生成过程都明确对应到论文符号。

## 1. 环境范围

PAV 只基于 RL4RS 的两个公开环境：

- `SlateRecEnv-v0`
- `SeqSlateRecEnv-v0`

第一版只使用 discrete action setting。也就是说，动作是 item id，而不是 continuous item embedding。continuous action 版本可以作为后续扩展。

## 2. RL4RS MDP 总体定义

RL4RS 中的推荐过程是有限时长 MDP：$M = (\mathcal{S}, \mathcal{A}, P, R, \gamma, H)$。

在第 $t$ 步，环境处于状态 $s_t$，推荐策略选择动作 $a_t$，simulator 根据动作更新状态并返回奖励。形式上：$s_{t+1} \sim P(\cdot \mid s_t, a_t)$，$r_t = R(s_t, a_t, s_{t+1})$。

策略目标是最大化 session 内折扣累计回报：$J(\pi) = \mathbb{E}_{\pi}[\sum_{t=0}^{H-1}\gamma^t r_t]$。

这套 MDP 不是我们额外假设的，而是 RL4RS 中 `RecEnvBase`、`SlateRecEnv`、`SeqSlateRecEnv`、`SlateState` 和 `SeqSlateState` 共同实现的。

## 3. Reset 与 episode 初始化

RL4RS 的 episode 从 `RecEnvBase.reset()` 开始。具体过程是：

1. `self.cur_step = 0`，环境步数归零。
2. `self.sim.reset(reset_file)` 重置数据缓存。
3. `self.samples, self.obs = self.sim.sample(self.batch_size)` 从 `RecDataBase` 中采样一批 logged records。
4. `sim.sample` 内部调用 `state_cls(config, records)`，也就是构造 `SlateState` 或 `SeqSlateState`。
5. 初始状态由 `records_to_state(records)` 从原始 record 解析得到。
6. `obs_fn(samples.state)` 将 simulator state 转成算法实际接收的 observation。

因此，初始状态 $s_0$ 不是随机生成的抽象变量，而是来自 RL4RS logged record 的用户/session 初始特征、历史行为、候选 item 信息和初始 mask。

## 4. 状态的精确定义

论文中可以将 RL4RS 状态写成：$s_t = (u, q_t, p_t, m_t, c_t)$。

其中：

- $u$ 表示用户或 session 初始特征，来自 record 中的 user profile 和 user sequence features。
- $q_t$ 表示当前 feature extraction 所需的 simulator state，包括用户序列特征、dense feature、category feature 等。
- $p_t = (a_0,\ldots,a_{t-1})$ 表示当前 slate 或当前 page 中已经推荐过的动作。
- $m_t$ 表示 action mask，用于屏蔽当前不能推荐的 item。
- $c_t$ 表示当前 step index，也就是源码中的 `cur_steps` 或 `cur_step`。

在 `SlateState` 中，状态核心变量包括：

- `self._init_state`：由 `records_to_state(records)` 得到的初始状态。
- `self._state`：当前 simulator state，会随着推荐动作更新。
- `self.prev_actions`：形状为 `(batch_size, max_steps)`，记录当前 episode 已经推荐过的 item。
- `self.action_mask`：屏蔽已经推荐过的 item。
- `self.special_mask`：屏蔽 special item 约束下不能再推荐的 item。
- `self.cur_steps`：当前 slate 内已经执行的推荐步数。

当 `support_d3rl_mask=True` 时，`SlateState.state` 返回一个字典，其中包括：

- `"state"`：当前内部 simulator state。
- `"masked_actions"`：已经推荐过的动作序列。
- `"cur_steps"`：当前 step index。

`SlateRecEnv.obs_fn` 再将这些内容转换成 d3rlpy 使用的 observation。默认 hidden observation 的拼接形式是：$\text{obs}_t = [z_t, \text{masked\_actions}_t, c_t]$，其中 $z_t$ 是 simulator 的 `obs_layer` 输出，维度通常为 256；`masked_actions` 长度为 9；`cur_steps` 长度为 1。因此默认离线数据中的 observation 维度为 $256 + 9 + 1$。

## 5. `SlateRecEnv-v0` 的状态更新

`SlateRecEnv-v0` 表示一个 9-item slate recommendation episode。默认 $H = 9$。

在第 $t$ 步，状态中已经保存了前面推荐过的 items：$p_t = (a_0,\ldots,a_{t-1})$。当策略选择动作 $a_t$ 后，`SlateState.act(actions)` 执行以下更新：

1. 如果是 continuous action，先通过 item embedding nearest neighbor 和 action mask 找到离散 item id；第一版 PAV 不使用此分支。
2. 将动作写入 `self.prev_actions[:, self.cur_steps] = actions`。
3. 将被推荐过的 item 在 `self.action_mask` 中置为 0，避免重复推荐。
4. 如果当前 slate 已经包含 special item，则用 `self.special_mask` 禁止继续推荐其他 special items。
5. 从 `self._init_state` 拷贝出临时状态 `tmp`。
6. 根据 `prev_actions` 取出所有已推荐 item 的 item vectors，并把这些 item vectors 与当前 item vector 拼到 dense feature 中。
7. 将 `sequence_id`、`prev_actions` 和当前 action 拼到 category feature 中。
8. 用更新后的 `tmp` 替换 `self._state`。
9. `self.cur_steps += 1`。

因此，`SlateRecEnv-v0` 的 transition 不只是简单追加 item id，而是同时更新推荐历史、mask、dense feature、category feature 和当前步数。论文中可以把它抽象为：$s_{t+1} = f_{\text{slate}}(s_t, a_t; {item_{info}}, {mask rules})$。

## 6. `SeqSlateRecEnv-v0` 的状态更新

`SeqSlateRecEnv-v0` 是多页 sequential slate setting。默认 `page_items = 9`，`max_steps = 36`，因此 $H = 36$，可以理解为连续 4 个 page。

它继承 `SlateState`，但在 `SeqSlateState.act(actions)` 中按 page 更新状态：

1. 将动作写入 `prev_actions[:, cur_steps]`。
2. 更新 `action_mask` 和 `special_mask`。
3. 计算当前 page 起止位置：`page_init = cur_steps // page_items * page_items`，`page_end = page_init + page_items - 1`。
4. `prev_expose` 只保留当前 page 之前已经曝光过的 items。
5. dense feature 只拼接当前 page 内的 item vectors 和当前 item vector。
6. category feature 拼接当前 page 的 `sequence_id`、当前 page 已曝光 items 和当前 action。
7. `cur_steps += 1`。
8. 如果 `cur_steps % page_items == 0`，说明一个 page 完成，重置 `action_mask` 和 `special_mask`，下一页可以重新开始选择 item。

因此，`SeqSlateRecEnv-v0` 的状态可以写成：$s_t = (u, \text{previous pages}, \text{current page prefix}, m_t, c_t)$。它的 transition 把推荐过程分成 page 内 item-level decisions 和 page boundary reward。

## 7. 动作空间与 action mask

第一版 PAV 使用离散动作空间：$a_t \in \{0,1,\ldots,|\mathcal{A}|-1\}$。

RL4RS 默认配置中，动作空间大小为 $|\mathcal{A}| = {action_{size}} = 284$。

动作表示推荐一个 item。环境通过 mask 控制可选动作：

- location mask：不同位置只能推荐特定范围内的 item。
- duplicate mask：同一 slate 或 page 内不能重复推荐相同 item。
- special item mask：special item 有额外互斥约束。

对于 discrete policy，算法输出 item id。对于 continuous policy，环境会通过 `get_nearest_neighbor_with_mask` 将 action embedding 映射到满足 mask 的最近 item id。第一版 PAV 不使用 continuous branch。

## 8. Observation 生成过程

RL4RS 的 observation 不是原始 state，而是 `obs_fn` 的输出。

`SlateRecEnv.obs_fn(state)` 的核心过程是：

1. 调用 `FeatureUtil.feature_extraction(state["state"])`，把内部 simulator state 转成模型输入特征。
2. 如果 `rawstate_as_obs=True`，直接返回 category、dense 和 sequence feature。
3. 默认情况下，将特征输入已经训练好的 simulator model 的 `simulator_obs` layer，得到 hidden observation $z_t$。
4. 如果 `support_d3rl_mask=True`，将 $z_t$、`masked_actions` 和 `cur_steps` 拼接成最终 observation。

因此，离线 RL 算法并不直接看到完整 record，而是看到经过 simulator observation layer 编码后的向量和 mask 相关信息。

## 9. Reward 的具体计算

RL4RS 同时存在 simulator reward 和 logged offline reward。PAV 主要基于离线 dataset 中的 reward 工作，因此要特别区分二者。

### 9.1 Simulator reward

`env.step(action)` 调用 `RecSimBase._step(samples, action, step=cur_step)`。具体过程是：

1. `samples.act(action)` 更新状态。
2. `next_state = samples.state`。
3. `next_obs = obs_fn(next_state)`。
4. `reward = forward(model, samples)`。
5. 如果当前 step 是最后一步，则 `done = 1`，否则 `done = 0`。

在 `SlateRecEnv.forward` 中，如果 slate 尚未完成，即 `samples.cur_steps < max_steps`，reward 为 0。只有 slate 完成后，才计算完整 slate reward：

1. 取出完整 `prev_actions`。
2. 构造 `complete_states`，即每个推荐位置对应的完整状态。
3. 用 `FeatureUtil.feature_extraction` 得到 simulator model 输入。
4. 通过 `reward_layer` 得到每个 item 的点击或成功概率。
5. 取每个 item 的 price。
6. 计算 slate reward：$r = \sum_i \text{price}_i \cdot p_i$。
7. 如果违反 mask 或约束，则 reward 置为 0。

在 `SeqSlateRecEnv.forward` 中，reward 只在 page boundary 计算。如果 `step % page_items != 0`，reward 为 0；如果一个 page 完成，则对最近一个 page 的 items 计算：$r = \sum_{i \in \text{current page}} \text{price}_i \cdot p_i$。

### 9.2 Logged offline reward

RL4RS 生成 offline dataset 时使用 `env.offline_reward` 写入 reward，而不是直接使用 `env.step(action)` 返回的 simulator reward。

在 `SlateState.offline_reward` 中，当前 step 小于 `max_steps` 时 reward 为 0；当完整 slate 结束后，读取 logged record 中的 exposed items 和 user feedback label，并计算：$r_{H-1} = \sum_i \text{price}_i \cdot \text{label}_i$。

在 `SeqSlateState.offline_reward` 中，如果当前 step 不是 9 的倍数，则 reward 为 0；如果刚完成一个 page，则取当前 page 的 actions、price 和 label，计算：$r_t = \sum_{i \in \text{page}} \text{price}_i \cdot \text{label}_i$。

这说明 RL4RS 的离线 reward 对 item-level action 是 delayed 的：单个 action 的影响通常要等到 slate 或 page 完成后才被记录。

## 10. Termination

`RecSimBase._step` 根据当前环境步数设置 done：

- 如果 `step < max_steps - 1`，则 `done = 0`。
- 如果 `step >= max_steps - 1`，则 `done = 1`。

因此，`SlateRecEnv-v0` 默认在第 9 个 item 决策后终止，`SeqSlateRecEnv-v0` 默认在第 36 个 item 决策后终止。

## 11. Offline Dataset 生成过程

RL4RS 的 offline RL 使用 d3rlpy 的 `MDPDataset`。源码中 `data_generate_rl4rs_a` 和 `data_generate_rl4rs_b` 会生成：

$\text{MDPDataset}(\text{observations}, \text{actions}, \text{rewards}, \text{terminals})$。

对于 `SlateRecEnv-v0`：

1. `obs = env.reset()` 得到初始 observation。
2. `action = env.offline_action` 读取 logged record 中第一个曝光 item。
3. 保存初始 observation、action、reward 0 和 terminal 0。
4. 循环 9 步：执行 `env.step(action)`，保存 next observation。
5. 再读取下一步 `env.offline_action` 作为 logged action。
6. 保存 `env.offline_reward` 作为 reward。
7. 保存 done。
8. 展平 batch 和 time 维度，构造 `MDPDataset`。

对于 `SeqSlateRecEnv-v0`，流程相同，但循环 `max_steps = 36` 步，reward 在每个 9-step page boundary 出现。

这意味着 PAV 的最自然插入点不是修改环境，而是在 `MDPDataset` 生成或加载之后，对每个 transition 计算 process signal，再构造 shaped reward。

## 12. Credit Assignment 问题

RL4RS 原始 reward 回答的是：完成的 slate、page 或 session 是否成功？

PAV 要回答的是更细粒度的问题：在状态 $s_t$ 下采取动作 $a_t$，是否推动 session 朝最终成功方向前进？

由于很多 reward 只有在完整 slate 或 page 结束后才出现，把同一个最终成功信号粗暴归因给前面所有 action 是不准确的。一个成功 slate 中可能同时包含关键动作、中性动作和有害动作。同样，一个本来就很好的状态也可能让某个动作看起来很好，即使该动作本身贡献很小。

因此，PAV 的目标是学习 process-level advantage，把状态本身的质量和动作带来的增量进步分开：

$\text{state quality}: s_t \text{ 本身已经有多好}$。

$\text{action progress}: a_t \text{ 从 } s_t \text{ 出发带来了多少增量进步}$。

后续 PAV 定义将直接建立在这个 RL4RS MDP 上。

