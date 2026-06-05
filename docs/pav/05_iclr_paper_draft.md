# Process Advantage Verifiers for Offline Reinforcement Learning in Slate Recommendation

## Abstract

Offline reinforcement learning for slate recommendation often receives reward only after a complete slate or page has been shown. This delayed feedback makes action-level credit assignment ambiguous: a high-return slate may contain helpful, neutral, and harmful item decisions. We propose Process Advantage Verifiers (PAV), a two-network framework that converts delayed RL4RS rewards into verified process-level reward shaping signals. A state-only Reward Model estimates state potential, non-network k-step progress measures local improvement over that state potential, and a Verifier predicts whether the local progress is reliable and aligned with the eventual return. The resulting verified contribution is added to the original reward and used to train standard offline RL algorithms.

## 1. Introduction

RL4RS provides realistic slate and sequential slate recommendation environments, but its offline reward is delayed: `SlateRecEnv-v0` mainly rewards a completed 9-item slate, while `SeqSlateRecEnv-v0` rewards completed pages. Standard offline RL methods such as CQL and BCQ therefore optimize from sparse process information. The central question is whether a process-level advantage signal can improve offline policy learning without redefining the RL4RS environment.

PAV addresses this by decomposing recommendation quality into state quality and action contribution. The Reward Model estimates how promising the current state already is. Progress then measures how much the trajectory improves after an action relative to that state potential. The Verifier filters progress that appears locally positive but is not consistent with final outcome improvement.

## 2. Preliminaries: RL4RS MDP

We use the public RL4RS MDP directly. The finite-horizon MDP is:

```text
M = (S, A, P, R, gamma, H)
```

The first version uses only discrete actions and two public environments:

- `SlateRecEnv-v0`, with default horizon `H = 9`.
- `SeqSlateRecEnv-v0`, with default horizon `H = 36`.

The state can be written as:

```text
s_t = (u, q_t, p_t, m_t, c_t)
```

where `u` is user/session context, `q_t` is the simulator state used for feature extraction, `p_t` is the prefix of recommended items, `m_t` is the action mask, and `c_t` is the current step index. The first PAV implementation operates after RL4RS `MDPDataset` generation, leaving the simulator and environment transition semantics unchanged.

## 3. Method

### Reward Model

The Reward Model is state-only:

```text
R_phi(s_t) ~= E[G_t | s_t]
```

It is trained with mean squared error against discounted returns:

```text
L_R = E[(R_phi(s_t) - G_t)^2]
```

This is intentionally not a Q-function because PAV needs to separate state quality from action contribution.

### Non-Network Progress

Progress is computed, not separately learned:

```text
p_t^k = sum_{i=0}^{K_t-1} gamma^i r_{t+i}
        + 1[K_t = k] gamma^k R_phi(s_{t+k})
        - R_phi(s_t)
```

where `K_t = min(k, H - t)`. This definition preserves page or slate rewards if they occur inside the k-step window, and otherwise falls back to state-potential improvement.

### Verifier

The Verifier estimates whether local progress is reliable:

```text
V_psi(s_t, a_t) = P(Z_t = 1 | s_t, a_t)
```

The first implementation uses an empirical outcome-consistency target:

```text
Z_t = 1[sign(p_t^k - b_p(t)) = sign(G_t - G_bar(t))]
```

where the baselines are computed by step or page position.

### Verified Contribution and Reward Shaping

The verified contribution is:

```text
C_t = p_t^k V_psi(s_t, a_t)
```

The shaped reward is:

```text
r'_t = r_t + alpha * clip(normalize(C_t), -3, 3)
```

The default settings are `alpha = 0.1`, `k = 3` for `SlateRecEnv-v0`, and `k = 5` for `SeqSlateRecEnv-v0`.

## 4. Experiments

The main comparison is:

```text
Offline RL(original reward) vs. Offline RL(verified-progress-shaped reward)
```

Algorithms:

- `CQL`
- `BCQ`
- `BC` as an auxiliary behavioral baseline

Metrics:

- simulator average episode reward
- reward mean and standard deviation
- offline policy evaluation: `CIPS`, `DR`, `WIPS`, `SeqDR`

Main table:

| Environment | Algorithm | PAV | Avg Reward | CIPS | DR | WIPS | SeqDR |
|---|---|---:|---:|---:|---:|---:|---:|
| SlateRecEnv-v0 | CQL | No |  |  |  |  |  |
| SlateRecEnv-v0 | CQL | Yes |  |  |  |  |  |
| SlateRecEnv-v0 | BCQ | No |  |  |  |  |  |
| SlateRecEnv-v0 | BCQ | Yes |  |  |  |  |  |
| SeqSlateRecEnv-v0 | CQL | No |  |  |  |  |  |
| SeqSlateRecEnv-v0 | CQL | Yes |  |  |  |  |  |

## 5. Ablations

Required ablations:

- `k in {1, 3, 5}`
- `alpha in {0.05, 0.1, 0.2}`
- learned `R_phi` vs. `R_phi = 0`
- verified contribution vs. raw progress
- Verifier gate vs. no Verifier gate
- with clipping vs. without clipping
- Reward Model and Verifier capacity: small, medium, large MLP

## 6. Mechanism Analysis

The mechanism analysis should show whether PAV supplies useful process signals before delayed reward arrives:

- step index vs. original reward
- step index vs. progress
- step index vs. verifier score
- step index vs. verified contribution
- step index vs. shaped reward
- `Z_t` label distribution

If performance does not improve, the analysis should identify whether the issue comes from target construction, reward scaling, verifier quality, or instability in offline RL training.

## 7. Limitations

The first version is restricted to RL4RS discrete actions and does not include continuous item embedding actions. It uses outcome consistency for `Z_t`; intent consistency based on category, label, or session targets remains future work. PAV also depends on the quality of the state-only Reward Model, so poor value estimates can produce noisy progress. Finally, reward shaping changes the offline learning signal and must be normalized carefully to avoid overwhelming the original RL4RS reward scale.
