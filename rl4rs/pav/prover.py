"""Explicit prover policies mu for Q^mu / MC rollouts."""
import os

import numpy as np
import torch
import torch.nn as nn

from rl4rs.pav.dataset import discrete_action_vector, ensure_dir
from rl4rs.pav.models import save_checkpoint
from rl4rs.online.env_utils import attach_slate_masks, sample_masked_actions, valid_action_mask


def _ensure_env_masks(env_config):
    if env_config is None:
        raise ValueError("env_config required for masked prover sampling.")
    if "location_mask" not in env_config:
        attach_slate_masks(env_config)
    return env_config


class PolicyScorer(nn.Module):
    """BC policy head: observation -> action logits (for supervised / bo_k provers)."""

    def __init__(self, observation_dim, action_size, hidden_units):
        super(PolicyScorer, self).__init__()
        layers = []
        last_dim = int(observation_dim)
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, int(action_size)))
        self.net = nn.Sequential(*layers)

    def forward(self, observations):
        return self.net(observations)


def _masked_logits(logits, obs, env_config):
    logits = np.asarray(logits, dtype=np.float64)
    mask = valid_action_mask(obs, env_config)
    logits[~mask] = -1e9
    return logits


def _sample_from_logits(logits, rng):
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(), 1e-8)
    return int(rng.choice(len(probs), p=probs))


class Prover(object):
    """Reference policy mu: maps state -> action distribution or sample."""

    kind = "base"

    def sample(self, observations, rng, env_config=None):
        raise NotImplementedError

    def describe(self):
        return {"prover_kind": self.kind}


class LoggingProver(Prover):
    """Default: use actions recorded in the offline flat dataset."""

    kind = "logging"

    def __init__(self, flat):
        self._actions = np.asarray(discrete_action_vector(flat["actions"]), dtype=np.int64)

    def sample(self, observations, rng, env_config=None):
        obs = np.asarray(observations)
        if obs.ndim == 1:
            raise ValueError("LoggingProver.sample expects batch indices via actions_at")
        return self._actions[: len(observations)]

    def actions_at(self, indices):
        return self._actions[np.asarray(indices, dtype=np.int64)]

    def describe(self):
        return {"prover_kind": self.kind}


class UniformProver(Prover):
    kind = "uniform"

    def sample(self, observations, rng, env_config):
        return sample_masked_actions(observations, rng, env_config)


class RandomMaskedProver(Prover):
    kind = "random"

    def sample(self, observations, rng, env_config):
        return sample_masked_actions(observations, rng, env_config)


class TorchPolicyProver(Prover):
    """Sample from a torch BC policy checkpoint (Policy_*.pt)."""

    kind = "torch_policy"

    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = str(checkpoint_path)
        self.device = device or torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        meta = checkpoint.get("metadata", {})
        obs_dim = int(meta.get("observation_dim", 266))
        action_size = int(meta.get("action_size", 284))
        hidden_units = meta.get("hidden_units", [256, 128])
        self.model = PolicyScorer(obs_dim, action_size, hidden_units).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def _logits_for_obs(self, obs):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)
        with torch.no_grad():
            return self.model(x).detach().cpu().numpy().reshape(-1)

    def sample(self, observations, rng, env_config):
        obs_list = observations if isinstance(observations, list) else [observations]
        actions = []
        for obs in obs_list:
            logits = _masked_logits(self._logits_for_obs(obs), obs, env_config)
            actions.append(_sample_from_logits(logits, rng))
        return actions

    def greedy(self, observations, env_config):
        if isinstance(observations, np.ndarray) and observations.ndim == 2:
            obs_list = [observations[i] for i in range(observations.shape[0])]
        else:
            obs_list = observations if isinstance(observations, list) else [observations]
        actions = []
        for obs in obs_list:
            logits = _masked_logits(self._logits_for_obs(obs), obs, env_config)
            actions.append(int(np.argmax(logits)))
        return actions


class SupervisedProver(TorchPolicyProver):
    kind = "supervised"

    def __init__(self, artifact_path, device=None):
        if not artifact_path or not os.path.isfile(artifact_path):
            raise FileNotFoundError(
                "SupervisedProver requires Policy checkpoint: {}".format(artifact_path)
            )
        super(SupervisedProver, self).__init__(artifact_path, device=device)

    def describe(self):
        return {"prover_kind": self.kind, "prover_artifact_path": self.checkpoint_path}


class BoKProver(TorchPolicyProver):
    """K masked samples + greedy re-rank by policy logits."""

    kind = "bo_k"

    def __init__(self, artifact_path, bo_k=3, device=None):
        if not artifact_path or not os.path.isfile(artifact_path):
            raise FileNotFoundError(
                "BoKProver requires Policy checkpoint: {}".format(artifact_path)
            )
        super(BoKProver, self).__init__(artifact_path, device=device)
        self.bo_k = max(1, int(bo_k))

    def sample(self, observations, rng, env_config):
        obs_list = observations if isinstance(observations, list) else [observations]
        actions = []
        for obs in obs_list:
            logits = _masked_logits(self._logits_for_obs(obs), obs, env_config)
            mask = valid_action_mask(obs, env_config)
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                raise RuntimeError("empty action mask for BoKProver")
            candidates = set()
            candidates.add(int(np.argmax(logits)))
            while len(candidates) < min(self.bo_k, len(valid)):
                candidates.add(int(rng.choice(valid)))
            best_action = max(candidates, key=lambda a: logits[a])
            actions.append(int(best_action))
        return actions

    def describe(self):
        return {
            "prover_kind": self.kind,
            "prover_bo_k": self.bo_k,
            "prover_artifact_path": self.checkpoint_path,
        }


def train_bc_policy_prover(flat, config, device=None):
    """Train BC policy on logged (obs, action) pairs; return TorchPolicyProver checkpoint path."""
    observations = np.asarray(flat["observations"], dtype=np.float32)
    actions = np.clip(discrete_action_vector(flat["actions"]), 0, config.action_size - 1)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PolicyScorer(observations.shape[1], config.action_size, config.hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    indices = np.arange(len(actions))
    if config.max_train_samples is not None and len(indices) > config.max_train_samples:
        indices = np.random.choice(indices, size=config.max_train_samples, replace=False)

    for epoch in range(max(2, min(config.verifier_epochs, 5))):
        model.train()
        losses = []
        perm = np.random.permutation(indices)
        for start in range(0, len(perm), config.batch_size):
            batch_idx = perm[start:start + config.batch_size]
            x = torch.as_tensor(observations[batch_idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(actions[batch_idx], dtype=torch.long, device=device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        print("PAV BC prover epoch {} ce {:.6f}".format(epoch + 1, float(np.mean(losses))))

    policy_path = os.path.join(
        config.pav_output_dir,
        "Policy_{}.pt".format(config.artifact_prefix),
    )
    ensure_dir(policy_path)
    save_checkpoint(policy_path, model, {
        "observation_dim": int(observations.shape[1]),
        "action_size": int(config.action_size),
        "hidden_units": config.hidden_units,
        "kind": "bc_policy",
    })
    return policy_path


def resolve_prover_artifact_path(config, flat=None, device=None):
    """Return checkpoint path for supervised/bo_k; train BC inline if missing."""
    path = str(getattr(config, "prover_artifact_path", "") or "")
    if path and os.path.isfile(path):
        return path
    if flat is None:
        raise ValueError("prover_artifact_path missing and flat dataset unavailable for BC training.")
    print("PAV: training inline BC policy prover (no artifact provided)", flush=True)
    return train_bc_policy_prover(flat, config, device=device)


def compute_prover_base_agreement(prover, observations, base_actions, env_config):
    """Fraction of states where prover greedy action matches base (DQN) action."""
    if prover is None or not hasattr(prover, "greedy"):
        return None
    obs = np.asarray(observations, dtype=np.float32)
    base = np.asarray(discrete_action_vector(base_actions), dtype=np.int64).reshape(-1)
    if len(obs) == 0:
        return None
    prover_actions = np.asarray(prover.greedy(obs, env_config), dtype=np.int64).reshape(-1)
    n = min(len(prover_actions), len(base))
    if n == 0:
        return None
    return float(np.mean(prover_actions[:n] == base[:n]))


def apply_prover_actions_to_flat(flat, prover, env_config, config, rng):
    """Replace flat actions with prover mu(a|s); keep logged actions for diagnostics."""
    flat = dict(flat)
    flat["logged_actions"] = np.asarray(flat["actions"]).copy()
    n = len(flat["observations"])
    indices = np.arange(n, dtype=np.int64)

    if isinstance(prover, LoggingProver):
        flat["actions"] = prover.actions_at(indices)
        flat["actions_source"] = "logging"
        return flat

    env_config = _ensure_env_masks(env_config)
    observations = flat["observations"]
    batch_size = max(1, int(getattr(config, "batch_size", 256)))
    sampled = []
    for start in range(0, n, batch_size):
        obs_chunk = observations[start:start + batch_size]
        obs_list = [obs_chunk[i] for i in range(len(obs_chunk))]
        actions = prover.sample(obs_list, rng, env_config)
        sampled.extend(int(a) for a in actions)
    flat["actions"] = np.asarray(sampled, dtype=np.int64).reshape(-1, 1)
    flat["actions_source"] = str(getattr(prover, "kind", "unknown"))
    return flat


def build_prover(config, flat=None, env_config=None, device=None):
    kind = str(getattr(config, "prover_kind", "logging") or "logging")
    rng = np.random.RandomState(int(getattr(config, "mc_seed", 0)))

    if kind == "logging":
        if flat is None:
            raise ValueError("LoggingProver requires flattened dataset.")
        return LoggingProver(flat), rng
    if kind == "uniform":
        if env_config is None:
            raise ValueError("UniformProver requires env_config with action masks.")
        return UniformProver(), rng
    if kind == "random":
        if env_config is None:
            raise ValueError("RandomMaskedProver requires env_config with action masks.")
        return RandomMaskedProver(), rng
    if kind == "supervised":
        artifact_path = resolve_prover_artifact_path(config, flat=flat, device=device)
        return SupervisedProver(artifact_path, device=device), rng
    if kind == "bo_k":
        artifact_path = resolve_prover_artifact_path(config, flat=flat, device=device)
        return BoKProver(artifact_path, config.prover_bo_k, device=device), rng
    raise ValueError("unknown prover_kind: {}".format(kind))


def prover_metadata(config, prover=None):
    meta = {
        "prover_kind": str(getattr(config, "prover_kind", "logging")),
        "prover_bo_k": int(getattr(config, "prover_bo_k", 1)),
        "prover_artifact_path": str(getattr(config, "prover_artifact_path", "") or ""),
    }
    if prover is not None and hasattr(prover, "describe"):
        meta.update(prover.describe())
    return meta


def format_prover_banner(stats_or_config):
    if isinstance(stats_or_config, dict):
        kind = stats_or_config.get("prover_kind", "logging")
        bo_k = stats_or_config.get("prover_bo_k", 1)
        path = stats_or_config.get("prover_artifact_path", "")
    else:
        kind = getattr(stats_or_config, "prover_kind", "logging")
        bo_k = getattr(stats_or_config, "prover_bo_k", 1)
        path = getattr(stats_or_config, "prover_artifact_path", "") or ""
    lines = ["PAV prover_kind = {}".format(kind)]
    if kind == "bo_k":
        lines.append("PAV prover_bo_k = {}".format(bo_k))
    if path:
        lines.append("PAV prover_artifact_path = {}".format(path))
    return lines
