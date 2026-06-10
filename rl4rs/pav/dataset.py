import os

import h5py
import numpy as np
from d3rlpy.dataset import MDPDataset
try:
    from d3rlpy.dataset import InfiniteBuffer
except ImportError:
    InfiniteBuffer = None


def ensure_dir(path):
    directory = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def load_mdpdataset(path):
    try:
        return MDPDataset.load(path)
    except TypeError:
        # d3rlpy>=2 changed ReplayBuffer.load signature and cannot read
        # legacy d3rlpy 0.x MDPDataset paths directly.
        try:
            with h5py.File(path, "r") as f:
                observations = f["observations"][:]
                actions = f["actions"][:]
                rewards = f["rewards"][:]
                terminals = f["terminals"][:]
                discrete_action = bool(f["discrete_action"][()]) if "discrete_action" in f else True
            kwargs = {}
            if not discrete_action:
                kwargs["action_space"] = None
            return MDPDataset(observations, actions, rewards, terminals, **kwargs)
        except (OSError, KeyError):
            if InfiniteBuffer is None:
                raise
            with open(path, "rb") as f:
                return MDPDataset.load(f, InfiniteBuffer())


def _episode_terminals(episode, length):
    terminals = getattr(episode, "terminals", None)
    if terminals is None:
        terminals = np.zeros((length,), dtype="float32")
        if getattr(episode, "terminal", True) and length:
            terminals[-1] = 1.0
    terminals = np.asarray(terminals, dtype="float32").reshape(-1)
    if terminals.shape[0] < length:
        padded = np.zeros((length,), dtype="float32")
        padded[:terminals.shape[0]] = terminals
        terminals = padded
    return terminals[:length]


def episode_to_arrays(episode):
    observations = np.asarray(episode.observations, dtype="float32")
    actions = np.asarray(episode.actions)
    rewards = np.asarray(episode.rewards, dtype="float32").reshape(-1)
    length = min(observations.shape[0], actions.shape[0], rewards.shape[0])
    observations = observations[:length]
    actions = actions[:length]
    rewards = rewards[:length]
    terminals = _episode_terminals(episode, length)
    return observations, actions, rewards, terminals


def iter_episode_arrays(dataset):
    for episode in dataset.episodes:
        observations, actions, rewards, terminals = episode_to_arrays(episode)
        if len(rewards) > 0:
            yield observations, actions, rewards, terminals


def flatten_episodes(dataset):
    obs_list, action_list, reward_list, terminal_list = [], [], [], []
    episode_ids, step_ids = [], []
    for episode_id, arrays in enumerate(iter_episode_arrays(dataset)):
        observations, actions, rewards, terminals = arrays
        length = len(rewards)
        obs_list.append(observations)
        action_list.append(actions)
        reward_list.append(rewards)
        terminal_list.append(terminals)
        episode_ids.append(np.full((length,), episode_id, dtype="int64"))
        step_ids.append(np.arange(length, dtype="int64"))

    if not obs_list:
        raise ValueError("MDPDataset contains no non-empty episodes.")

    return {
        "observations": np.concatenate(obs_list, axis=0).astype("float32"),
        "actions": np.concatenate(action_list, axis=0),
        "rewards": np.concatenate(reward_list, axis=0).astype("float32"),
        "terminals": np.concatenate(terminal_list, axis=0).astype("float32"),
        "episode_ids": np.concatenate(episode_ids, axis=0),
        "step_ids": np.concatenate(step_ids, axis=0),
    }


def discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype="float32")
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return returns


def add_returns(flat, gamma):
    returns = np.zeros_like(flat["rewards"], dtype="float32")
    for episode_id in np.unique(flat["episode_ids"]):
        idx = np.where(flat["episode_ids"] == episode_id)[0]
        returns[idx] = discounted_returns(flat["rewards"][idx], gamma)
    flat["returns"] = returns
    return flat


def export_mdpdataset(observations, actions, rewards, terminals, path, discrete_action=True):
    ensure_dir(path)
    actions = np.asarray(actions)
    if discrete_action and actions.ndim == 2 and actions.shape[1] == 1:
        actions = actions[:, 0]
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=observations.astype("float32"))
        f.create_dataset("actions", data=actions.astype("int32") if discrete_action else actions.astype("float32"))
        f.create_dataset("rewards", data=rewards.astype("float32"))
        f.create_dataset("terminals", data=terminals.astype("float32"))
        f.create_dataset("episode_terminals", data=terminals.astype("float32"))
        f.create_dataset("discrete_action", data=bool(discrete_action))
        f.create_dataset("create_mask", data=False)
        f.create_dataset("mask_size", data=0)
    return path


def is_discrete_actions(actions):
    actions = np.asarray(actions)
    return actions.ndim == 1 or (actions.ndim == 2 and actions.shape[1] == 1)


def discrete_action_vector(actions):
    actions = np.asarray(actions)
    if actions.ndim == 2 and actions.shape[1] == 1:
        actions = actions[:, 0]
    if actions.ndim != 1:
        raise ValueError("PAV v1 supports discrete actions only.")
    return actions.astype("int64")
