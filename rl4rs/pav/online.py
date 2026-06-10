import json
import os

import numpy as np
import torch

from rl4rs.online.env_utils import as_batch_list, batch_pav_obs_vectors_from_env, obs_vector
from rl4rs.pav.config import PAVConfig
from rl4rs.pav.models import load_reward_model, load_verifier
from rl4rs.pav.progress import (
    combine_progress,
    compute_directional_progress,
    compute_k_step_progress,
    normalize_by_step,
)
from rl4rs.pav.trainer import predict_reward_embeddings, predict_reward_values, predict_verifier_scores


def _device(config):
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _norm_stats_for_config(stats, config):
    """Pick frozen normalization stats matching contribution definition."""
    if config.use_raw_progress or not config.use_verifier:
        return stats.get("progress_norm_by_step") or stats.get("norm_by_step", {})
    return stats.get("norm_by_step", {})


def normalize_with_frozen_stats(values, step_ids, norm_by_step, eps=1e-6):
    values = np.asarray(values, dtype=np.float32)
    step_ids = np.asarray(step_ids, dtype=np.int64)
    normalized = np.zeros_like(values, dtype=np.float32)
    for step_id in np.unique(step_ids):
        idx = step_ids == step_id
        stats = norm_by_step.get(str(int(step_id)), {"mean": 0.0, "std": 1.0})
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0))
        if std < eps:
            std = 1.0
        normalized[idx] = (values[idx] - mean) / std
    return normalized


def shape_rewards_frozen(
    raw_rewards,
    contribution,
    step_ids,
    norm_by_step,
    alpha,
    clip_c,
    use_clipping,
):
    normalized = normalize_with_frozen_stats(contribution, step_ids, norm_by_step)
    if use_clipping:
        normalized = np.clip(normalized, -clip_c, clip_c)
    raw = np.asarray(raw_rewards, dtype=np.float32)
    shaped = raw + alpha * normalized.astype(np.float32)
    return shaped.astype(np.float32), normalized.astype(np.float32)


def _flat_from_sequences(observations, actions, rewards):
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    length = len(rewards)
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "step_ids": np.arange(length, dtype=np.int64),
        "episode_ids": np.zeros(length, dtype=np.int64),
    }


def _transitions_to_flat(transitions):
    return _flat_from_sequences(
        [obs_vector(t["obs"]) for t in transitions],
        [int(t["action"]) for t in transitions],
        [float(t["raw_reward"]) for t in transitions],
    )


def _compute_progress_array(flat, artifacts):
    config = artifacts["config"]
    values = predict_reward_values(artifacts["reward_model"], flat["observations"], config)
    potential = compute_k_step_progress(flat, values, config.k, config.gamma)
    if float(config.directional_lambda) > 0.0:
        embeddings = predict_reward_embeddings(
            artifacts["reward_model"], flat["observations"], config
        )
        directional = compute_directional_progress(flat, embeddings, config.k)
        return combine_progress(potential, directional, config.directional_lambda)
    return potential


def _contribution_array(flat, progress, artifacts):
    config = artifacts["config"]
    if config.use_raw_progress or not config.use_verifier or artifacts["verifier"] is None:
        return np.asarray(progress, dtype=np.float32)
    actions = np.clip(flat["actions"], 0, config.action_size - 1)
    verifier_scores = predict_verifier_scores(
        artifacts["verifier"], flat["observations"], actions, config
    )
    return np.asarray(progress, dtype=np.float32) * verifier_scores


def compute_shaped_rewards_for_flat(flat, artifacts):
    """Full v2 progress shaping for every step in flat."""
    config = artifacts["config"]
    stats = artifacts["stats"]
    progress = _compute_progress_array(flat, artifacts)
    contribution = _contribution_array(flat, progress, artifacts)
    norm_by_step = _norm_stats_for_config(stats, config)
    return shape_rewards_frozen(
        flat["rewards"],
        contribution,
        flat["step_ids"],
        norm_by_step,
        alpha=config.alpha,
        clip_c=config.clip_c,
        use_clipping=config.use_clipping,
    )[0]


def _shape_latest_from_flat(flat, artifacts):
    if len(flat["rewards"]) == 0:
        return 0.0
    shaped = compute_shaped_rewards_for_flat(flat, artifacts)
    return float(shaped[-1])


class _EpisodeBuffer(object):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def append(self, observation, action, reward):
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))

    def to_flat(self):
        if not self.rewards:
            return None
        return _flat_from_sequences(self.observations, self.actions, self.rewards)


def fit_pav_models(dataset, config):
    """Train PAV reward/verifier checkpoints."""
    from rl4rs.pav.trainer import build_pav_signals

    if isinstance(config, dict):
        config = PAVConfig.from_dict(config)
    return build_pav_signals(dataset, config)


def _sync_online_config(config, stats):
    """Apply checkpoint stats (k, alpha, directional_lambda, clip). Verifier gate from config/CLI."""
    config.k = int(stats.get("k", config.k))
    config.alpha = float(stats.get("alpha", config.alpha))
    config.clip_c = float(stats.get("clip_c", config.clip_c))
    config.directional_lambda = float(stats.get("directional_lambda", config.directional_lambda))
    config.use_clipping = bool(stats.get("use_clipping", config.use_clipping))
    return config


def load_pav_artifacts(config_or_dict):
    """Load frozen PAV checkpoints + stats json for online inference."""
    config = config_or_dict if isinstance(config_or_dict, PAVConfig) else PAVConfig.from_dict(config_or_dict)
    device = _device(config)

    if not os.path.isfile(config.stats_path):
        raise FileNotFoundError("PAV stats not found: {}".format(config.stats_path))
    with open(config.stats_path, "r") as f:
        stats = json.load(f)

    config = _sync_online_config(config, stats)

    reward_ckpt = torch.load(config.reward_model_path, map_location=device)
    reward_meta = reward_ckpt.get("metadata", {})
    obs_dim = int(reward_meta.get("observation_dim", 266))
    hidden_units = reward_meta.get("hidden_units", config.hidden_units)
    reward_model, _ = load_reward_model(
        config.reward_model_path,
        obs_dim,
        hidden_units,
        device,
    )

    verifier = None
    if config.use_verifier and os.path.isfile(config.verifier_path):
        verifier_ckpt = torch.load(config.verifier_path, map_location=device)
        ver_meta = verifier_ckpt.get("metadata", {})
        verifier, _ = load_verifier(
            config.verifier_path,
            int(ver_meta.get("observation_dim", obs_dim)),
            int(ver_meta.get("action_size", config.action_size)),
            ver_meta.get("hidden_units", hidden_units),
            device,
        )

    return {
        "config": config,
        "stats": stats,
        "reward_model": reward_model,
        "verifier": verifier,
        "device": device,
    }


def shape_latest_steps_batched(observations, actions, raw_rewards, step_ids, artifacts, episode_buffers=None):
    """Shaped reward for the latest step in a vector env (full v2 progress via episode prefix)."""
    config = artifacts["config"]
    raw = np.asarray(raw_rewards, dtype=np.float32)
    batch_size = len(raw)
    shaped = np.zeros(batch_size, dtype=np.float32)

    if episode_buffers is None:
        obs = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        step_ids = np.asarray(step_ids, dtype=np.int64)
        for i in range(batch_size):
            flat = _flat_from_sequences(
                obs[i:i + 1], actions[i:i + 1], raw[i:i + 1]
            )
            shaped[i] = _shape_latest_from_flat(flat, artifacts)
        return shaped.astype(np.float32)

    for i in range(batch_size):
        flat = episode_buffers[i].to_flat()
        if flat is None:
            shaped[i] = float(raw[i])
        else:
            shaped[i] = _shape_latest_from_flat(flat, artifacts)
    return shaped.astype(np.float32)


def apply_pav_to_episode(transitions, artifacts):
    """Dense per-step shaped rewards for one episode."""
    if not transitions:
        return []
    flat = _transitions_to_flat(transitions)
    return compute_shaped_rewards_for_flat(flat, artifacts).tolist()


def compute_step_shaped_reward(transitions_prefix, artifacts):
    """Shaped reward for the latest step (incremental API)."""
    if not transitions_prefix:
        return 0.0
    flat = _transitions_to_flat(transitions_prefix)
    return _shape_latest_from_flat(flat, artifacts)


class PAVRewardWrapper(object):
    """Gym wrapper: full v2 progress (k-step + directional) × Verifier gate each step."""

    def __init__(self, env, artifacts=None, enabled=True):
        self.env = env
        self.artifacts = artifacts
        self.enabled = bool(enabled and artifacts is not None)
        self.batch_size = int(getattr(env, "batch_size", 1))
        self._step_ids = [0 for _ in range(self.batch_size)]
        self._buffers = [_EpisodeBuffer() for _ in range(self.batch_size)]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self, **kwargs):
        self._step_ids = [0 for _ in range(self.batch_size)]
        for buf in self._buffers:
            buf.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        pre_obs = self.env.state
        obs_batch = batch_pav_obs_vectors_from_env(self.env, pre_obs)
        next_obs, raw_reward, done, info = self.env.step(action)

        if not self.enabled:
            return next_obs, raw_reward, done, info

        actions = np.asarray(as_batch_list(action, self.batch_size), dtype=np.int64)
        raw_items = np.asarray(as_batch_list(raw_reward, self.batch_size), dtype=np.float32)
        done_items = as_batch_list(done, self.batch_size)

        for i in range(self.batch_size):
            self._buffers[i].append(obs_batch[i], actions[i], raw_items[i])

        step_id_arr = np.asarray(self._step_ids, dtype=np.int64)
        shaped_items = shape_latest_steps_batched(
            obs_batch, actions, raw_items, step_id_arr, self.artifacts, self._buffers
        )

        for i, dn in enumerate(done_items):
            self._step_ids[i] += 1
            if dn:
                self._step_ids[i] = 0
                self._buffers[i].clear()

        if self.batch_size == 1:
            return next_obs, float(shaped_items[0]), done, info
        return next_obs, shaped_items.tolist(), done, info
