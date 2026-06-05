import numpy as np


def compute_k_step_progress(flat, state_values, k, gamma):
    """Compute non-network k-step progress inside each episode."""
    rewards = flat["rewards"]
    episode_ids = flat["episode_ids"]
    progress = np.zeros_like(rewards, dtype="float32")

    for episode_id in np.unique(episode_ids):
        idx = np.where(episode_ids == episode_id)[0]
        ep_rewards = rewards[idx]
        ep_values = state_values[idx]
        length = len(idx)

        for local_t in range(length):
            horizon = min(k, length - local_t)
            reward_sum = 0.0
            for i in range(horizon):
                reward_sum += (gamma ** i) * float(ep_rewards[local_t + i])

            bootstrap = 0.0
            if horizon == k and local_t + k < length:
                bootstrap = (gamma ** k) * float(ep_values[local_t + k])

            progress[idx[local_t]] = reward_sum + bootstrap - float(ep_values[local_t])

    return progress.astype("float32")


def step_baselines(values, step_ids):
    values = np.asarray(values, dtype="float32")
    step_ids = np.asarray(step_ids, dtype="int64")
    baselines = np.zeros_like(values, dtype="float32")
    for step_id in np.unique(step_ids):
        idx = step_ids == step_id
        baselines[idx] = float(np.mean(values[idx]))
    return baselines


def verifier_labels(progress, returns, step_ids):
    progress_baseline = step_baselines(progress, step_ids)
    return_baseline = step_baselines(returns, step_ids)
    progress_sign = np.sign(progress - progress_baseline)
    return_sign = np.sign(returns - return_baseline)
    labels = (progress_sign == return_sign).astype("float32")
    return labels, progress_baseline, return_baseline


def normalize_by_step(values, step_ids, eps=1e-6):
    values = np.asarray(values, dtype="float32")
    step_ids = np.asarray(step_ids, dtype="int64")
    normalized = np.zeros_like(values, dtype="float32")
    stats = {}
    for step_id in np.unique(step_ids):
        idx = step_ids == step_id
        mean = float(np.mean(values[idx]))
        std = float(np.std(values[idx]))
        if std < eps:
            std = 1.0
        normalized[idx] = (values[idx] - mean) / std
        stats[str(int(step_id))] = {"mean": mean, "std": std}
    return normalized, stats


def shape_rewards(rewards, contribution, step_ids, alpha=0.1, clip_c=3.0, use_clipping=True):
    normalized, stats = normalize_by_step(contribution, step_ids)
    if use_clipping:
        normalized = np.clip(normalized, -clip_c, clip_c)
    shaped = rewards.astype("float32") + alpha * normalized.astype("float32")
    return shaped.astype("float32"), normalized.astype("float32"), stats
