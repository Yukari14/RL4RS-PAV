import json
import os

import numpy as np
import torch

from rl4rs.online.env_utils import as_batch_list, batch_pav_obs_vectors_from_env, obs_vector
from rl4rs.pav.config import PAVConfig
from rl4rs.pav.models import load_reward_model, load_verifier
from rl4rs.pav.progress import compute_k_step_progress
from rl4rs.pav.trainer import predict_reward_values, predict_verifier_scores


def _device(config):
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    alpha_scale=None,
):
    normalized = normalize_with_frozen_stats(contribution, step_ids, norm_by_step)
    if use_clipping:
        normalized = np.clip(normalized, -clip_c, clip_c)
    raw = np.asarray(raw_rewards, dtype=np.float32)
    if alpha_scale is None:
        shaped = raw + alpha * normalized.astype(np.float32)
    else:
        scale = np.asarray(alpha_scale, dtype=np.float32)
        shaped = raw + alpha * scale * normalized.astype(np.float32)
    return shaped.astype(np.float32), normalized.astype(np.float32)


def verifier_confidence_scores(verifier_scores):
    """Map verifier prob in [0,1] to confidence in [0,1] (1 = most confident)."""
    scores = np.asarray(verifier_scores, dtype=np.float32)
    return np.abs(scores - 0.5) * 2.0


def compute_alpha_scale(verifier_scores, config):
    """Per-sample multiplier for alpha when confidence_gating is enabled."""
    if not config.confidence_gating:
        return None
    confidence = verifier_confidence_scores(verifier_scores)
    floor = float(config.min_confidence)
    return np.maximum(confidence, floor).astype(np.float32)


def cap_shaped_rewards(raw_rewards, shaped_rewards, config):
    """Limit shaping term magnitude so it cannot dominate sparse raw rewards."""
    ratio = float(config.max_shaping_ratio)
    if ratio <= 0:
        return shaped_rewards
    raw = np.asarray(raw_rewards, dtype=np.float32)
    shaped = np.asarray(shaped_rewards, dtype=np.float32)
    floor = float(config.shaping_abs_floor)
    max_delta = ratio * np.maximum(np.abs(raw), floor)
    delta = shaped - raw
    delta = np.clip(delta, -max_delta, max_delta)
    return (raw + delta).astype(np.float32)


def _transitions_to_flat(transitions):
    observations = np.stack([obs_vector(t["obs"]) for t in transitions], axis=0).astype(np.float32)
    actions = np.asarray([int(t["action"]) for t in transitions], dtype=np.int64)
    rewards = np.asarray([float(t["raw_reward"]) for t in transitions], dtype=np.float32)
    step_ids = np.arange(len(transitions), dtype=np.int64)
    episode_ids = np.zeros(len(transitions), dtype=np.int64)
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "step_ids": step_ids,
        "episode_ids": episode_ids,
    }


def fit_pav_models(dataset, config):
    """Train reward/verifier checkpoints (offline fit). Returns build_pav_signals bundle."""
    from rl4rs.pav.trainer import build_pav_signals

    if isinstance(config, dict):
        config = PAVConfig.from_dict(config)
    return build_pav_signals(dataset, config)


def load_pav_artifacts(config_or_dict):
    """Load frozen PAV checkpoints + stats json for online inference."""
    config = config_or_dict if isinstance(config_or_dict, PAVConfig) else PAVConfig.from_dict(config_or_dict)
    device = _device(config)

    if not os.path.isfile(config.stats_path):
        raise FileNotFoundError("PAV stats not found: {}".format(config.stats_path))
    with open(config.stats_path, "r") as f:
        stats = json.load(f)

    reward_ckpt = torch.load(config.reward_model_path, map_location=device)
    reward_meta = reward_ckpt.get("metadata", {})
    obs_dim = int(reward_meta.get("observation_dim", 256))
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


def shape_latest_steps_batched(observations, actions, raw_rewards, step_ids, artifacts):
    """Online Scheme C: batched shaped reward for the current step only.

    When rewards are returned step-by-step during rollouts, the latest transition
    always has horizon=min(k, 1)=1 in k-step progress, i.e. progress = raw - V(s).
    This matches taking shaped[-1] from prefix replay but avoids O(t) recomputation.
    """
    config = artifacts["config"]
    stats = artifacts["stats"]
    obs = np.asarray(observations, dtype=np.float32)
    raw = np.asarray(raw_rewards, dtype=np.float32)
    step_ids = np.asarray(step_ids, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)

    values = predict_reward_values(artifacts["reward_model"], obs, config)
    progress = raw - values

    if config.use_verifier and artifacts["verifier"] is not None:
        actions_clipped = np.clip(actions, 0, config.action_size - 1)
        verifier_scores = predict_verifier_scores(
            artifacts["verifier"], obs, actions_clipped, config
        )
    else:
        verifier_scores = np.ones_like(progress, dtype=np.float32)

    contribution = progress if config.use_raw_progress else progress * verifier_scores
    alpha_scale = compute_alpha_scale(verifier_scores, config)
    shaped, _normalized = shape_rewards_frozen(
        raw,
        contribution,
        step_ids,
        stats.get("norm_by_step", {}),
        alpha=config.alpha,
        clip_c=config.clip_c,
        use_clipping=config.use_clipping,
        alpha_scale=alpha_scale,
    )
    shaped = cap_shaped_rewards(raw, shaped, config)
    return shaped.astype(np.float32)


def apply_pav_to_episode(transitions, artifacts):
    """Dense per-step shaped rewards for one episode (offline / batch helper)."""
    if not transitions:
        return []

    config = artifacts["config"]
    stats = artifacts["stats"]
    flat = _transitions_to_flat(transitions)

    values = predict_reward_values(artifacts["reward_model"], flat["observations"], config)
    progress = compute_k_step_progress(flat, values, config.k, config.gamma)

    if config.use_verifier and artifacts["verifier"] is not None:
        actions = np.clip(flat["actions"], 0, config.action_size - 1)
        verifier_scores = predict_verifier_scores(
            artifacts["verifier"], flat["observations"], actions, config
        )
    else:
        verifier_scores = np.ones_like(progress, dtype=np.float32)

    if config.use_raw_progress:
        contribution = progress
    else:
        contribution = progress * verifier_scores

    alpha_scale = compute_alpha_scale(verifier_scores, config)
    shaped, _normalized = shape_rewards_frozen(
        flat["rewards"],
        contribution,
        flat["step_ids"],
        stats.get("norm_by_step", {}),
        alpha=config.alpha,
        clip_c=config.clip_c,
        use_clipping=config.use_clipping,
        alpha_scale=alpha_scale,
    )
    shaped = cap_shaped_rewards(flat["rewards"], shaped, config)
    return shaped.tolist()


def compute_step_shaped_reward(transitions_prefix, artifacts):
    """Shaped reward for the latest step (single-slot API, incremental)."""
    if not transitions_prefix:
        return 0.0
    last = transitions_prefix[-1]
    step_id = len(transitions_prefix) - 1
    obs = np.asarray([obs_vector(last["obs"])], dtype=np.float32)
    actions = np.asarray([int(last["action"])], dtype=np.int64)
    raw = np.asarray([float(last["raw_reward"])], dtype=np.float32)
    step_ids = np.asarray([step_id], dtype=np.int64)
    shaped = shape_latest_steps_batched(obs, actions, raw, step_ids, artifacts)
    return float(shaped[0])


class PAVRewardWrapper(object):
    """Gym-style wrapper: streaming dense PAV shaped reward each step (Scheme C)."""

    def __init__(self, env, artifacts=None, enabled=True):
        self.env = env
        self.artifacts = artifacts
        self.enabled = bool(enabled and artifacts is not None)
        self.batch_size = int(getattr(env, "batch_size", 1))
        self._step_ids = [0 for _ in range(self.batch_size)]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self, **kwargs):
        self._step_ids = [0 for _ in range(self.batch_size)]
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
        step_id_arr = np.asarray(self._step_ids, dtype=np.int64)

        shaped_items = shape_latest_steps_batched(
            obs_batch, actions, raw_items, step_id_arr, self.artifacts
        )

        for i, dn in enumerate(done_items):
            self._step_ids[i] += 1
            if dn:
                self._step_ids[i] = 0

        if self.batch_size == 1:
            return next_obs, float(shaped_items[0]), done, info
        return next_obs, shaped_items.tolist(), done, info
