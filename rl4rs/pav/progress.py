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


def _cosine_similarity(a, b, eps=1e-8):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denom = np.maximum(denom, eps)
    return np.sum(a * b, axis=-1) / denom


def compute_directional_progress(flat, embeddings, k):
    """Directional progress: cos(e_{t+K}, g) - cos(e_t, g), g = embedding at episode start."""
    embeddings = np.asarray(embeddings, dtype="float32")
    episode_ids = flat["episode_ids"]
    directional = np.zeros((len(embeddings),), dtype="float32")

    for episode_id in np.unique(episode_ids):
        idx = np.where(episode_ids == episode_id)[0]
        ep_embeddings = embeddings[idx]
        goal = ep_embeddings[0]
        length = len(idx)

        for local_t in range(length):
            k_t = min(k, length - 1 - local_t)
            if k_t <= 0:
                directional[idx[local_t]] = 0.0
                continue
            future_local = local_t + k_t
            cos_future = float(_cosine_similarity(ep_embeddings[future_local], goal))
            cos_current = float(_cosine_similarity(ep_embeddings[local_t], goal))
            directional[idx[local_t]] = cos_future - cos_current

    return directional.astype("float32")


def combine_progress(potential_progress, directional_progress, directional_lambda):
    potential_progress = np.asarray(potential_progress, dtype="float32")
    if directional_lambda == 0.0 or directional_progress is None:
        return potential_progress.astype("float32")
    directional_progress = np.asarray(directional_progress, dtype="float32")
    return (potential_progress + directional_lambda * directional_progress).astype("float32")


def step_baselines(values, step_ids):
    values = np.asarray(values, dtype="float32")
    step_ids = np.asarray(step_ids, dtype="int64")
    baselines = np.zeros_like(values, dtype="float32")
    for step_id in np.unique(step_ids):
        idx = step_ids == step_id
        baselines[idx] = float(np.mean(values[idx]))
    return baselines


def _step_margin(values, step_ids, margin_frac):
    margins = np.zeros_like(values, dtype="float32")
    for step_id in np.unique(step_ids):
        idx = step_ids == step_id
        std = float(np.std(values[idx]))
        if std < 1e-6:
            std = 1.0
        margins[idx] = margin_frac * std
    return margins


def verifier_labels(progress, returns, step_ids, mode="sign", margin_frac=0.25,
                    state_values=None):
    """Build verifier targets.

    mode:
      sign / magnitude — legacy outcome-consistency labels
      necessity — proxy N_t = G_t - R_phi(s_t) above step baseline
      necessity_combined — necessity AND positive progress excess
    """
    mode = str(mode)
    if mode in ("necessity", "necessity_combined"):
        if state_values is None:
            raise ValueError("state_values required for necessity verifier labels.")
        return necessity_verifier_labels(
            progress, returns, state_values, step_ids,
            mode=mode, margin_frac=margin_frac,
        )

    progress_baseline = step_baselines(progress, step_ids)
    return_baseline = step_baselines(returns, step_ids)
    prog_excess = progress - progress_baseline
    ret_excess = returns - return_baseline
    progress_sign = np.sign(prog_excess)
    return_sign = np.sign(ret_excess)
    same_sign = (progress_sign == return_sign).astype("float32")

    if mode == "sign":
        labels = same_sign
    elif mode == "magnitude":
        labels = np.zeros_like(same_sign, dtype="float32")
        for step_id in np.unique(step_ids):
            idx = step_ids == step_id
            ret_std = float(np.std(ret_excess[idx]))
            if ret_std < 1e-6:
                ret_std = 1.0
            margin = margin_frac * ret_std
            strong = np.abs(ret_excess[idx]) >= margin
            labels[idx] = same_sign[idx] * strong.astype("float32")
    else:
        raise ValueError("unknown verifier_label_mode: {}".format(mode))

    return labels.astype("float32"), progress_baseline, return_baseline


def necessity_verifier_labels(progress, returns, state_values, step_ids,
                              mode="necessity", margin_frac=0.25):
    """Proxy necessity labels without counterfactual rollouts."""
    progress = np.asarray(progress, dtype="float32")
    returns = np.asarray(returns, dtype="float32")
    state_values = np.asarray(state_values, dtype="float32")
    step_ids = np.asarray(step_ids, dtype="int64")

    necessity = returns - state_values
    necessity_baseline = step_baselines(necessity, step_ids)
    necessity_excess = necessity - necessity_baseline
    margins = _step_margin(necessity_excess, step_ids, margin_frac)
    labels = (necessity_excess > margins).astype("float32")

    if mode == "necessity_combined":
        progress_baseline = step_baselines(progress, step_ids)
        progress_excess = progress - progress_baseline
        prog_margins = _step_margin(progress_excess, step_ids, margin_frac)
        labels = labels * (progress_excess > prog_margins).astype("float32")

    progress_baseline = step_baselines(progress, step_ids)
    return_baseline = step_baselines(returns, step_ids)
    return labels.astype("float32"), progress_baseline, return_baseline


def build_progress_pair_indices(flat, k):
    """Indices for differentiable k-step progress: (t_idx, tk_idx, reward_sum, bootstrap_mask)."""
    rewards = flat["rewards"]
    episode_ids = flat["episode_ids"]
    t_indices = []
    tk_indices = []
    reward_sums = []
    bootstrap_masks = []

    for episode_id in np.unique(episode_ids):
        idx = np.where(episode_ids == episode_id)[0]
        ep_rewards = rewards[idx]
        length = len(idx)

        for local_t in range(length):
            horizon = min(k, length - local_t)
            reward_sum = 0.0
            for i in range(horizon):
                reward_sum += float(ep_rewards[local_t + i])

            bootstrap_mask = 1.0 if horizon == k and local_t + k < length else 0.0
            tk_local = local_t + horizon if bootstrap_mask > 0 else local_t

            t_indices.append(int(idx[local_t]))
            tk_indices.append(int(idx[tk_local]))
            reward_sums.append(reward_sum)
            bootstrap_masks.append(bootstrap_mask)

    return {
        "t_indices": np.asarray(t_indices, dtype="int64"),
        "tk_indices": np.asarray(tk_indices, dtype="int64"),
        "reward_sums": np.asarray(reward_sums, dtype="float32"),
        "bootstrap_masks": np.asarray(bootstrap_masks, dtype="float32"),
    }


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
