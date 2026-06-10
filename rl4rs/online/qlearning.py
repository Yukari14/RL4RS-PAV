import os

import numpy as np
import torch
import torch.nn as nn

from rl4rs.online.env_utils import obs_list, obs_vector, sample_masked_actions, valid_action_mask


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_units=(256, 128)):
        super(QNetwork, self).__init__()
        layers = []
        last = obs_dim
        for h in hidden_units:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def masked_argmax(q_values, action_mask):
    q = np.asarray(q_values, dtype=np.float64).copy()
    mask = np.asarray(action_mask, dtype=bool)
    q[~mask] = -np.inf
    return int(np.argmax(q))


def _batch_obs_vectors(obs):
    return np.stack([obs_vector(o) for o in obs_list(obs)], axis=0).astype(np.float32)


def _batch_masks(obs, env_config):
    return np.stack([valid_action_mask(o, env_config) for o in obs_list(obs)], axis=0)


def select_actions_batch(q_net, obs, epsilon, rng, device, env_config):
    items = obs_list(obs)
    obs_batch = _batch_obs_vectors(obs)
    masks = _batch_masks(obs, env_config)
    actions = [None] * len(items)

    greedy_idx = []
    with torch.no_grad():
        q_batch = q_net(torch.as_tensor(obs_batch, dtype=torch.float32, device=device)).cpu().numpy()

    for i, o in enumerate(items):
        if rng.rand() < epsilon:
            actions[i] = sample_masked_actions(o, rng, env_config)[0]
        else:
            actions[i] = masked_argmax(q_batch[i], masks[i])
    return actions


def masked_max_q_batch(q_net, obs, device, env_config):
    obs_batch = _batch_obs_vectors(obs)
    masks = _batch_masks(obs, env_config)
    with torch.no_grad():
        q_batch = q_net(torch.as_tensor(obs_batch, dtype=torch.float32, device=device)).cpu().numpy()
    max_q = []
    for i in range(len(obs_list(obs))):
        valid = np.flatnonzero(masks[i])
        if len(valid) == 0:
            max_q.append(0.0)
        else:
            max_q.append(float(np.max(q_batch[i][valid])))
    return max_q


def collect_episodes_batch(env, q_net, epsilon, rng, device, max_steps, env_config):
    batch_size = int(env.batch_size)
    obs = env.reset(reset_file=False)
    trajectories = [[] for _ in range(batch_size)]
    train_returns = np.zeros(batch_size, dtype=np.float64)

    for _step in range(max_steps):
        actions = select_actions_batch(q_net, obs, epsilon, rng, device, env_config)
        next_obs, rewards, dones, _info = env.step(actions)
        reward_list = list(rewards) if isinstance(rewards, (list, tuple, np.ndarray)) else [rewards]
        done_list = list(dones) if isinstance(dones, (list, tuple, np.ndarray)) else [dones]
        pre_items = obs_list(obs)
        next_items = obs_list(next_obs)

        for i in range(batch_size):
            r = float(reward_list[i])
            trajectories[i].append({
                "obs": pre_items[i],
                "action": int(actions[i]),
                "reward": r,
                "next_obs": next_items[i],
                "done": bool(done_list[i]),
            })
            train_returns[i] += r
        obs = next_obs

    raw_terminals = [float(traj[-1]["reward"]) if traj else 0.0 for traj in trajectories]
    return trajectories, raw_terminals, train_returns


def td_update_batch(q_net, optimizer, trajectories, gamma, device, env_config):
    losses = []
    for traj in trajectories:
        for trans in traj:
            obs_t = torch.as_tensor(obs_vector(trans["obs"]), dtype=torch.float32, device=device)
            action = int(trans["action"])
            target = float(trans["reward"])
            if not trans["done"]:
                target += gamma * masked_max_q_batch(
                    q_net, trans["next_obs"], device, env_config
                )[0]

            q_all = q_net(obs_t.unsqueeze(0)).squeeze(0)
            q_sa = q_all[action]
            loss = (q_sa - target) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    return losses


def eval_greedy(env, q_net, device, max_steps, n_episodes, env_config, seed=0):
    rng = np.random.RandomState(seed)
    env.seed(seed)
    batch_size = max(int(env.batch_size), 1)
    returns = []
    actions = []
    episodes_done = 0

    while episodes_done < n_episodes:
        obs = env.reset(reset_file=(episodes_done == 0))
        ep_returns = np.zeros(batch_size, dtype=np.float64)
        for _step in range(max_steps):
            step_actions = select_actions_batch(
                q_net, obs, epsilon=0.0, rng=rng, device=device, env_config=env_config
            )
            actions.extend(step_actions)
            obs, rewards, dones, _info = env.step(step_actions)
            reward_list = list(rewards) if isinstance(rewards, (list, tuple, np.ndarray)) else [rewards]
            ep_returns += np.asarray(reward_list, dtype=np.float64)
        returns.extend(ep_returns.tolist())
        episodes_done += batch_size

    returns = np.asarray(returns[:n_episodes], dtype=np.float64)
    actions = np.asarray(actions[: n_episodes * max_steps], dtype=np.int64)
    vals, counts = np.unique(actions, return_counts=True)
    order = np.argsort(counts)[::-1]
    diversity = {
        "unique_actions": int(len(vals)),
        "top_action_rate": float(counts[order[0]] / max(len(actions), 1)),
    }
    return {
        "sim_avg_reward": float(returns.mean()),
        "sim_std_reward": float(returns.std()),
        "action_diversity_masked": diversity,
    }


def train_qlearning(
    env,
    q_net,
    optimizer,
    num_episodes,
    max_steps,
    gamma,
    epsilon_start,
    epsilon_end,
    device,
    env_config,
    seed=0,
    log_every=50,
):
    rng = np.random.RandomState(seed)
    env.seed(seed)
    batch_size = max(int(env.batch_size), 1)
    num_batches = (num_episodes + batch_size - 1) // batch_size
    history = []

    for batch_idx in range(num_batches):
        frac = batch_idx / max(num_batches - 1, 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac
        trajectories, raw_terms, train_rets = collect_episodes_batch(
            env, q_net, epsilon, rng, device, max_steps, env_config
        )
        losses = td_update_batch(q_net, optimizer, trajectories, gamma, device, env_config)
        episodes = min((batch_idx + 1) * batch_size, num_episodes)
        if episodes % log_every == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
            msg = "ep {} eps {:.3f} train_ret {:.2f} td_loss {:.4f} batch={}".format(
                episodes,
                epsilon,
                float(np.mean(train_rets)),
                float(np.mean(losses)),
                batch_size,
            )
            print(msg, flush=True)
            history.append({
                "episode": episodes,
                "epsilon": epsilon,
                "train_return": float(np.mean(train_rets)),
                "raw_terminal_proxy": float(np.mean(raw_terms)),
                "td_loss": float(np.mean(losses)),
                "batch_size": batch_size,
            })
    return history


def save_q_checkpoint(path, q_net, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": q_net.state_dict(), "metadata": metadata}, path)


def load_q_checkpoint(path, obs_dim, action_dim, hidden_units, device):
    q_net = QNetwork(obs_dim, action_dim, hidden_units).to(device)
    payload = torch.load(path, map_location=device)
    q_net.load_state_dict(payload["model"])
    q_net.eval()
    return q_net, payload.get("metadata", {})
