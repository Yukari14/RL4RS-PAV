import json
import os
from collections import deque
from datetime import datetime

import numpy as np
import torch

from rl4rs.online.env_utils import as_batch_list, batch_pav_obs_vectors_from_env, obs_vector
from rl4rs.pav.config import PAVConfig
from rl4rs.pav.diagnostics import format_prover_quality, online_rolling_distinguishability
from rl4rs.pav.models import load_reward_model, load_verifier
from rl4rs.pav.progress import (
    combine_progress,
    compute_directional_progress,
    compute_k_step_progress,
    normalize_by_step,
)
from rl4rs.pav.prover import format_prover_banner
from rl4rs.pav.trainer import (
    predict_reward_embeddings,
    predict_reward_values,
    predict_verifier_q,
    predict_verifier_scores,
)


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
    normalize_contribution=True,
):
    if normalize_contribution:
        normalized = normalize_with_frozen_stats(contribution, step_ids, norm_by_step)
    else:
        normalized = np.asarray(contribution, dtype=np.float32)
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
    output_mode = str(getattr(config, "verifier_output_mode", "binary"))
    if output_mode in ("q_regression", "dual"):
        values = predict_reward_values(artifacts["reward_model"], flat["observations"], config)
        q_sa = predict_verifier_q(artifacts["verifier"], flat["observations"], actions, config)
        advantage = q_sa - values
        if output_mode == "dual":
            gate = predict_verifier_scores(
                artifacts["verifier"], flat["observations"], actions, config
            )
            return np.asarray(advantage, dtype=np.float32) * gate
        return np.asarray(advantage, dtype=np.float32)
    verifier_scores = predict_verifier_scores(
        artifacts["verifier"], flat["observations"], actions, config
    )
    return np.asarray(progress, dtype=np.float32) * verifier_scores


def compute_shaped_rewards_for_flat(flat, artifacts, alpha_scale=1.0):
    """Full v2 progress shaping for every step in flat."""
    config = artifacts["config"]
    stats = artifacts["stats"]
    progress = _compute_progress_array(flat, artifacts)
    contribution = _contribution_array(flat, progress, artifacts)
    norm_by_step = _norm_stats_for_config(stats, config)
    effective_alpha = float(config.alpha) * float(alpha_scale)
    return shape_rewards_frozen(
        flat["rewards"],
        contribution,
        flat["step_ids"],
        norm_by_step,
        alpha=effective_alpha,
        clip_c=config.clip_c,
        use_clipping=config.use_clipping,
        normalize_contribution=getattr(config, "normalize_contribution", False),
    )[0]


def _shape_latest_from_flat(flat, artifacts, alpha_scale=1.0):
    if len(flat["rewards"]) == 0:
        return 0.0
    shaped = compute_shaped_rewards_for_flat(flat, artifacts, alpha_scale=alpha_scale)
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
    config.normalize_contribution = bool(
        stats.get("normalize_contribution", getattr(config, "normalize_contribution", True))
    )
    config.verifier_output_mode = str(
        stats.get("verifier_output_mode", getattr(config, "verifier_output_mode", "binary"))
    )
    config.prover_kind = str(stats.get("prover_kind", getattr(config, "prover_kind", "logging")))
    config.distinguishability_floor = float(
        stats.get("distinguishability_floor", getattr(config, "distinguishability_floor", 0.05))
    )
    config.alpha_decay_enabled = bool(
        stats.get("alpha_decay_enabled", getattr(config, "alpha_decay_enabled", True))
    )
    config.alpha_decay_rate = float(
        stats.get("alpha_decay_rate", getattr(config, "alpha_decay_rate", 0.1))
    )
    return config


def _load_offline_prover(config, stats, device):
    """Load offline-pretrained prover (fixed during RL). logging needs no runtime object."""
    from rl4rs.pav.prover import BoKProver, SupervisedProver

    kind = str(stats.get("prover_kind", getattr(config, "prover_kind", "logging")))
    if kind == "logging":
        return None
    path = str(stats.get("prover_artifact_path", "") or getattr(config, "prover_artifact_path", "") or "")
    if kind in ("supervised", "bo_k") and not path:
        path = os.path.join(
            config.pav_output_dir,
            "Policy_{}.pt".format(config.artifact_prefix),
        )
    if kind == "supervised":
        if not os.path.isfile(path):
            print(
                "[PAV] supervised prover checkpoint missing (prover_kind={}): {}".format(kind, path),
                flush=True,
            )
            return None
        return SupervisedProver(path, device=device)
    if kind == "bo_k":
        if not os.path.isfile(path):
            print(
                "[PAV] bo_k prover checkpoint missing (prover_kind={}): {}".format(kind, path),
                flush=True,
            )
            return None
        return BoKProver(path, int(getattr(config, "prover_bo_k", 3)), device=device)
    return None


def load_pav_artifacts(config_or_dict):
    """Load frozen PAV checkpoints + stats json for online inference."""
    config = config_or_dict if isinstance(config_or_dict, PAVConfig) else PAVConfig.from_dict(config_or_dict)
    device = _device(config)

    if not os.path.isfile(config.stats_path):
        raise FileNotFoundError("PAV stats not found: {}".format(config.stats_path))
    with open(config.stats_path, "r") as f:
        stats = json.load(f)

    config = _sync_online_config(config, stats)
    prover = _load_offline_prover(config, stats, device)

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
        "prover": prover,
        "slate_env_config": None,
    }


def shape_latest_steps_batched(observations, actions, raw_rewards, step_ids, artifacts, episode_buffers=None, alpha_scale=1.0):
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
            shaped[i] = _shape_latest_from_flat(flat, artifacts, alpha_scale=alpha_scale)
        return shaped.astype(np.float32)

    for i in range(batch_size):
        flat = episode_buffers[i].to_flat()
        if flat is None:
            shaped[i] = float(raw[i])
        else:
            shaped[i] = _shape_latest_from_flat(flat, artifacts, alpha_scale=alpha_scale)
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

    def __init__(self, env, artifacts=None, enabled=True, monitor_every=None,
                 monitor_log_path=None, monitor_state_path=None):
        self.env = env
        self.artifacts = artifacts
        self.enabled = bool(enabled and artifacts is not None)
        self.batch_size = int(getattr(env, "batch_size", 1))
        self._step_ids = [0 for _ in range(self.batch_size)]
        self._buffers = [_EpisodeBuffer() for _ in range(self.batch_size)]
        self._monitor_log_path = monitor_log_path
        self._monitor_state_path = monitor_state_path
        if monitor_log_path:
            os.makedirs(os.path.dirname(monitor_log_path) or ".", exist_ok=True)
            if not os.path.isfile(monitor_log_path):
                with open(monitor_log_path, "w", newline="") as f:
                    import csv
                    csv.writer(f).writerow([
                        "timestamp", "transitions", "rolling_distinguishability",
                        "mean_verifier", "prover_agreement_rolling", "alpha_scale",
                    ])
        config = artifacts["config"] if artifacts is not None else None
        every = monitor_every
        if every is None and config is not None:
            every = int(getattr(config, "monitor_every", 1000))
        self._monitor_every = int(every or 0)
        self._monitor_progress = deque(maxlen=self._monitor_every or 1)
        self._monitor_contribution = deque(maxlen=self._monitor_every or 1)
        self._monitor_step_ids = deque(maxlen=self._monitor_every or 1)
        self._monitor_verifier = deque(maxlen=self._monitor_every or 1)
        self._monitor_prover_agree = deque(maxlen=self._monitor_every or 1)
        self._global_step = 0
        self._alpha_scale = 1.0
        if config is not None:
            self._alpha_scale = 1.0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self, **kwargs):
        self._step_ids = [0 for _ in range(self.batch_size)]
        for buf in self._buffers:
            buf.clear()
        return self.env.reset(**kwargs)

    def _flush_monitor(self):
        if self._monitor_every <= 0:
            return
        if self._global_step % self._monitor_every != 0:
            return
        if len(self._monitor_contribution) < 2 or len(self._monitor_step_ids) < 2:
            return
        dist, _ = online_rolling_distinguishability(
            np.asarray(self._monitor_contribution),
            np.asarray(self._monitor_step_ids, dtype=np.int64),
            min_count=2,
        )
        mean_verifier = float(np.mean(self._monitor_verifier)) if self._monitor_verifier else None
        rolling_agree = None
        if self._monitor_prover_agree:
            rolling_agree = float(np.mean(self._monitor_prover_agree))
        config = self.artifacts["config"]
        floor = float(getattr(config, "distinguishability_floor", 0.05))
        if getattr(config, "alpha_decay_enabled", True) and dist is not None and dist < floor:
            rate = float(getattr(config, "alpha_decay_rate", 0.1))
            self._alpha_scale = max(0.0, self._alpha_scale * (1.0 - rate))
        ts = datetime.now().isoformat()
        stats = self.artifacts.get("stats") or {}
        prover_kind = stats.get(
            "prover_kind",
            getattr(self.artifacts["config"], "prover_kind", "logging"),
        )
        print(
            "[PAV monitor] transitions={} rolling_distinguishability={} mean_verifier={} "
            "prover_kind={} prover_agree={} alpha_scale={:.3f}".format(
                self._global_step,
                "{:.4f}".format(dist) if dist is not None else "n/a",
                "{:.4f}".format(mean_verifier) if mean_verifier is not None else "n/a",
                prover_kind,
                "{:.4f}".format(rolling_agree) if rolling_agree is not None else "n/a",
                self._alpha_scale,
            ),
            flush=True,
        )
        if self._monitor_log_path:
            import csv
            with open(self._monitor_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts,
                    self._global_step,
                    dist if dist is not None else "",
                    mean_verifier if mean_verifier is not None else "",
                    rolling_agree if rolling_agree is not None else "",
                    self._alpha_scale,
                ])
        if self._monitor_state_path:
            import json
            stats = self.artifacts.get("stats") or {}
            payload = {
                "updated_at": ts,
                "transitions": int(self._global_step),
                "rolling_distinguishability": dist,
                "rolling_distinguishability_mode": "step_stratified_online",
                "offline_distinguishability": stats.get("distinguishability"),
                "offline_alignment_corr": stats.get("alignment_corr"),
                "prover_kind": stats.get(
                    "prover_kind",
                    getattr(self.artifacts["config"], "prover_kind", "logging"),
                ),
                "prover_agreement_rolling": rolling_agree,
                "mean_verifier": mean_verifier,
                "alpha_scale": float(self._alpha_scale),
            }
            with open(self._monitor_state_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

    def _record_monitor_batch(self, monitor_flats):
        if self._monitor_every <= 0 or not monitor_flats:
            return
        for flat in monitor_flats:
            progress = _compute_progress_array(flat, self.artifacts)
            contribution = _contribution_array(flat, progress, self.artifacts)
            step_ids = np.asarray(flat["step_ids"], dtype=np.int64).reshape(-1)
            self._monitor_progress.extend(progress.tolist())
            self._monitor_contribution.extend(np.asarray(contribution, dtype=np.float32).tolist())
            self._monitor_step_ids.extend(step_ids.tolist())
            if self.artifacts["config"].use_verifier and self.artifacts.get("verifier") is not None:
                act = np.clip(flat["actions"], 0, self.artifacts["config"].action_size - 1)
                scores = predict_verifier_scores(
                    self.artifacts["verifier"], flat["observations"], act, self.artifacts["config"]
                )
                self._monitor_verifier.extend(scores.tolist())
            prover = self.artifacts.get("prover")
            env_cfg = self.artifacts.get("slate_env_config")
            if prover is not None and env_cfg is not None and hasattr(prover, "greedy"):
                from rl4rs.pav.prover import compute_prover_base_agreement
                agree = compute_prover_base_agreement(
                    prover, flat["observations"], flat["actions"], env_cfg
                )
                if agree is not None:
                    self._monitor_prover_agree.extend([agree])
        self._global_step += self.batch_size
        self._flush_monitor()

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
            obs_batch, actions, raw_items, step_id_arr, self.artifacts, self._buffers,
            alpha_scale=self._alpha_scale,
        )

        if self._monitor_every > 0:
            monitor_flats = []
            for i in range(self.batch_size):
                flat = self._buffers[i].to_flat()
                if flat is not None:
                    monitor_flats.append(flat)
            if monitor_flats:
                self._record_monitor_batch(monitor_flats)

        for i, dn in enumerate(done_items):
            self._step_ids[i] += 1
            if dn:
                self._step_ids[i] = 0
                self._buffers[i].clear()

        if self.batch_size == 1:
            return next_obs, float(shaped_items[0]), done, info
        return next_obs, shaped_items.tolist(), done, info


def print_pav_artifact_summary(artifacts, pav_config=None):
    """Print offline artifact diagnostics visible at online training startup."""
    stats = artifacts.get("stats", {}) or {}
    cfg = pav_config or artifacts.get("config")
    print("", flush=True)
    print("=" * 72, flush=True)
    print("PAV ARTIFACT SUMMARY (offline-trained, frozen during DQN)", flush=True)
    print("=" * 72, flush=True)
    for line in format_prover_banner(stats if stats else cfg):
        print("  " + line, flush=True)
    fields = [
        ("actions_source", stats.get("actions_source")),
        ("verifier_output_mode", stats.get("verifier_output_mode", getattr(cfg, "verifier_output_mode", None))),
        ("reward_target", stats.get("reward_target", getattr(cfg, "reward_target", None))),
        ("normalize_contribution", stats.get("normalize_contribution")),
        ("use_hybrid_mc", stats.get("use_hybrid_mc")),
        ("use_trajectory_q_avg", stats.get("use_trajectory_q_avg")),
        ("alpha (shaping)", stats.get("alpha", getattr(cfg, "alpha", None))),
        ("contribution_return_corr", stats.get("contribution_return_corr")),
        ("value_mse", (stats.get("reward_metrics") or {}).get("value_mse")),
        ("verifier_q_mse", (stats.get("verifier_metrics") or {}).get("verifier_q_mse")),
        ("verifier_auc", (stats.get("verifier_metrics") or {}).get("verifier_auc")),
    ]
    for key, val in fields:
        if val is not None:
            print("  {} = {}".format(key, val), flush=True)
    for line in format_prover_quality(stats):
        print(line, flush=True)
    print("=" * 72, flush=True)
    print("", flush=True)


def print_pav_online_startup(artifacts, monitor_state_path=None):
    """Startup banner for online pilots (§5)."""
    stats = artifacts.get("stats", {})
    cfg = artifacts["config"]
    prover_kind = stats.get("prover_kind", getattr(cfg, "prover_kind", "logging"))
    print("PAV prover_kind = {} (frozen offline; reward + verifier also frozen)".format(prover_kind), flush=True)
    for line in format_prover_banner(stats):
        if not line.startswith("PAV prover_kind"):
            print(line, flush=True)
    print("Initial base policy ≈ random init", flush=True)
    if monitor_state_path:
        print("  PAV online state file: {}".format(monitor_state_path), flush=True)
    print(
        "Rolling monitor every {} transitions: distinguishability, verifier, prover_kind.".format(
            getattr(cfg, "monitor_every", 500)
        ),
        flush=True,
    )
