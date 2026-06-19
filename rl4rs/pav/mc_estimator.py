"""Monte Carlo Q^mu targets for verifier regression (improvement plan §3)."""
import threading

import numpy as np

from rl4rs.pav.dataset import add_returns, discrete_action_vector, flatten_episodes
from rl4rs.pav.prover import LoggingProver

_MC_PROGRESS = {"done": 0, "total": 0, "every": 25}
_MC_PROGRESS_LOCK = threading.Lock()


def _env_reset(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def _env_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, term, trunc, info = result
        done = bool(term or trunc)
        return obs, reward, done, info
    obs, reward, done, info = result
    return obs, reward, bool(done), info


def restore_from_prefix(env, prefix_actions, gamma=1.0):
    """Replay prefix from env.reset(); return obs, cumulative reward, steps taken, done."""
    obs = _env_reset(env)
    total = 0.0
    discount = 1.0
    steps = 0
    for action in prefix_actions:
        obs, reward, done, _info = _env_step(env, int(action))
        total += discount * float(reward)
        discount *= gamma
        steps += 1
        if done:
            return obs, total, steps, True
    return obs, total, steps, False


def _episode_slices(flat):
    terminals = np.asarray(flat["terminals"], dtype=np.float32).reshape(-1)
    starts = [0]
    for idx, term in enumerate(terminals):
        if term > 0.5:
            starts.append(idx + 1)
    starts.append(len(terminals))
    slices = []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        if e > s:
            slices.append((s, e))
    return slices


def estimate_q_mu_trajectory_average(flat, gamma=1.0):
    """Group (step_id, action) and average logged returns as Q proxy."""
    actions = discrete_action_vector(flat["actions"])
    returns = np.asarray(flat["returns"], dtype=np.float32).reshape(-1)
    step_ids = np.asarray(flat["step_ids"], dtype=np.int64).reshape(-1)
    keys = np.stack([step_ids, actions], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros(len(uniq), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.float64)
    for i, g in enumerate(inv):
        sums[g] += returns[i]
        counts[g] += 1.0
    means = sums / np.maximum(counts, 1.0)
    return means[inv].astype(np.float32)


def _select_mc_indices(flat, config):
    n = len(flat["returns"])
    max_states = int(getattr(config, "max_mc_states", 5000) or n)
    n_cov = int(getattr(config, "n_cov", -1))
    rng = np.random.RandomState(int(getattr(config, "mc_seed", 0)))
    if n_cov > 0 and n_cov < n:
        return np.sort(rng.choice(n, size=n_cov, replace=False))
    if max_states > 0 and max_states < n:
        return np.sort(rng.choice(n, size=max_states, replace=False))
    return np.arange(n, dtype=np.int64)


def _episode_for_index(flat_idx, slices):
    for s, e in slices:
        if s <= flat_idx < e:
            return s, e, flat_idx - s
    return None


def _complete_with_prover(env, prover, env_config, ep_start, ep_end, actions, step, gamma, rng):
    """Continue rollout from current env state using prover until episode ends."""
    from rl4rs.online.env_utils import pav_obs_vector_from_env, valid_action_mask

    total = 0.0
    discount = 1.0
    done = False
    obs_vec = pav_obs_vector_from_env(env, 0)
    mask = valid_action_mask(obs_vec, env_config)
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0.0

    while not done:
        if isinstance(prover, LoggingProver):
            if step < ep_end - ep_start:
                action = int(actions[ep_start + step])
            else:
                action = int(rng.choice(valid))
        elif hasattr(prover, "sample"):
            sample_obs = obs_vec if obs_vec.ndim == 1 else obs_vec[0]
            action = int(prover.sample([sample_obs], rng, env_config)[0])
        else:
            action = int(rng.choice(valid))
        _obs, reward, done, _info = _env_step(env, action)
        total += discount * float(reward)
        discount *= gamma
        step += 1
        if not done:
            obs_vec = pav_obs_vector_from_env(env, 0)
            mask = valid_action_mask(obs_vec, env_config)
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break
    return total


def _estimate_q_single_index(flat_idx, flat, config, prover, env_config, env, slices, actions):
    ep_info = _episode_for_index(flat_idx, slices)
    if ep_info is None:
        return flat_idx, None
    ep_start, ep_end, t = ep_info
    prefix_actions = actions[ep_start: ep_start + t + 1]
    n_mc = int(getattr(config, "n_mc", 8))
    gamma = float(getattr(config, "gamma", 1.0))
    rng = np.random.RandomState(int(getattr(config, "mc_seed", 0)) + int(flat_idx))

    mc_returns = []
    for _ in range(n_mc):
        _obs, prefix_return, _steps, done = restore_from_prefix(env, prefix_actions, gamma=gamma)
        if done:
            mc_returns.append(prefix_return)
            continue
        tail = _complete_with_prover(
            env, prover, env_config, ep_start, ep_end, actions, len(prefix_actions), gamma, rng
        )
        mc_returns.append(prefix_return + tail)
    if not mc_returns:
        return flat_idx, None
    return flat_idx, float(np.mean(mc_returns))


def _chunk_indices(indices, n_chunks):
    indices = np.asarray(indices, dtype=np.int64)
    if n_chunks <= 1 or len(indices) <= 1:
        return [indices]
    n_chunks = min(n_chunks, len(indices))
    return np.array_split(indices, n_chunks)


def _process_mc_chunk(flat_indices, flat, config, prover, env_config, env_factory, slices, actions):
    """One simulator env per worker chunk (reuse across many states)."""
    env = env_factory(env_config)
    results = []
    try:
        for flat_idx in flat_indices:
            idx, value = _estimate_q_single_index(
                int(flat_idx), flat, config, prover, env_config, env, slices, actions
            )
            results.append((idx, value))
            _tick_mc_progress(config)
    finally:
        if hasattr(env, "close"):
            env.close()
    return results


def _print_mc_progress(completed, total):
    print(
        "PAV hybrid MC progress {}/{} ({:.0f}%)".format(
            completed, total, 100.0 * completed / max(total, 1)
        ),
        flush=True,
    )


def _maybe_print_mc_progress(config):
    with _MC_PROGRESS_LOCK:
        done = _MC_PROGRESS["done"]
        total = _MC_PROGRESS["total"]
        every = _MC_PROGRESS["every"]
    if done == total or done % max(1, every) == 0:
        _print_mc_progress(done, total)


def _tick_mc_progress(config, n=1):
    with _MC_PROGRESS_LOCK:
        _MC_PROGRESS["done"] += int(n)
    _maybe_print_mc_progress(config)


def _init_mc_progress(total, config):
    every = int(getattr(config, "mc_progress_every", 25) or 25)
    with _MC_PROGRESS_LOCK:
        _MC_PROGRESS["done"] = 0
        _MC_PROGRESS["total"] = int(total)
        _MC_PROGRESS["every"] = max(1, every)


def estimate_q_mu_simulator(flat, config, prover, env_config, env_factory, indices=None):
    """Replay logged prefixes via restore_from_prefix; complete with prover for n_mc rollouts."""
    if indices is None:
        indices = _select_mc_indices(flat, config)
    q_mu = np.asarray(flat["returns"], dtype=np.float32).copy()
    actions = discrete_action_vector(flat["actions"])
    slices = _episode_slices(flat)
    indices = np.asarray(indices, dtype=np.int64)
    max_workers = int(getattr(config, "mc_max_workers", 1) or 1)
    total = len(indices)
    n_mc = int(getattr(config, "n_mc", 8))
    sim_cpu = bool(getattr(config, "mc_sim_use_cpu", False))
    sim_batch = int(getattr(config, "mc_sim_batch_size", 1) or 1)
    _init_mc_progress(total, config)
    print(
        "PAV hybrid MC: {} states x {} rollouts, {} workers, sim={} batch={} (env reuse)".format(
            total,
            n_mc,
            max(1, min(max_workers, total)),
            "cpu" if sim_cpu else "gpu",
            sim_batch,
        ),
        flush=True,
    )

    if max_workers <= 1 or total <= 1:
        env = env_factory(env_config)
        try:
            for flat_idx in indices:
                idx, value = _estimate_q_single_index(
                    int(flat_idx), flat, config, prover, env_config, env, slices, actions
                )
                if value is not None:
                    q_mu[idx] = value
                _tick_mc_progress(config)
        finally:
            if hasattr(env, "close"):
                env.close()
        return q_mu

    from concurrent.futures import ThreadPoolExecutor, as_completed

    chunks = _chunk_indices(indices, max_workers)
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(
                _process_mc_chunk,
                chunk,
                flat,
                config,
                prover,
                env_config,
                env_factory,
                slices,
                actions,
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            for idx, value in future.result():
                if value is not None:
                    q_mu[idx] = value
    return q_mu


def _select_hybrid_indices(flat, config, value_baseline):
    """Pick high-uncertainty states for simulator MC (hybrid mode)."""
    baseline = np.asarray(value_baseline, dtype=np.float32).reshape(-1)
    returns = np.asarray(flat["returns"], dtype=np.float32).reshape(-1)
    uncertainty = np.abs(returns - baseline)
    max_states = int(getattr(config, "max_mc_states", 5000) or len(returns))
    fraction = float(getattr(config, "hybrid_mc_fraction", 0.2))
    n_pick = max(1, min(max_states, int(len(returns) * fraction)))
    order = np.argsort(-uncertainty)
    return np.sort(order[:n_pick])


def estimate_q_mu(flat, config, prover=None, env_config=None, env_factory=None, value_baseline=None):
    """Main entry: trajectory average base + optional simulator / hybrid MC refinement."""
    if getattr(config, "use_trajectory_q_avg", True):
        base = estimate_q_mu_trajectory_average(flat, gamma=config.gamma)
    else:
        base = np.asarray(flat["returns"], dtype=np.float32).reshape(-1)

    use_sim = bool(getattr(config, "use_simulator_q", False))
    use_hybrid = bool(getattr(config, "use_hybrid_mc", False))
    if not use_sim and not use_hybrid:
        return base.astype(np.float32)

    if env_config is None or env_factory is None:
        if use_sim:
            raise ValueError("use_simulator_q requires env_config and env_factory.")
        return base.astype(np.float32)

    indices = None
    if use_hybrid and value_baseline is not None:
        indices = _select_hybrid_indices(flat, config, value_baseline)
    elif not use_sim:
        return base.astype(np.float32)

    q_mu = base.astype(np.float32).copy()
    sim_values = estimate_q_mu_simulator(
        flat, config, prover, env_config, env_factory, indices=indices
    )
    if indices is None:
        return sim_values
    q_mu[indices] = sim_values[indices]
    return q_mu


def mc_metadata(config):
    return {
        "use_simulator_q": bool(getattr(config, "use_simulator_q", False)),
        "use_trajectory_q_avg": bool(getattr(config, "use_trajectory_q_avg", True)),
        "use_hybrid_mc": bool(getattr(config, "use_hybrid_mc", False)),
        "hybrid_mc_fraction": float(getattr(config, "hybrid_mc_fraction", 0.2)),
        "reward_target": str(getattr(config, "reward_target", "q_mu")),
        "n_mc": int(getattr(config, "n_mc", 8)),
        "n_cov": int(getattr(config, "n_cov", -1)),
        "mc_seed": int(getattr(config, "mc_seed", 0)),
        "max_mc_states": int(getattr(config, "max_mc_states", 5000)),
        "mc_max_workers": int(getattr(config, "mc_max_workers", 4)),
        "mc_sim_batch_size": int(getattr(config, "mc_sim_batch_size", 1)),
        "mc_sim_use_cpu": bool(getattr(config, "mc_sim_use_cpu", False)),
        "mc_progress_every": int(getattr(config, "mc_progress_every", 25)),
    }


def build_flat_with_q_mu(dataset, config, prover=None, env_config=None, env_factory=None):
    flat = flatten_episodes(dataset)
    flat = add_returns(flat, config.gamma)
    q_mu = estimate_q_mu(flat, config, prover=prover, env_config=env_config, env_factory=env_factory)
    flat = dict(flat)
    flat["q_mu"] = q_mu
    return flat
