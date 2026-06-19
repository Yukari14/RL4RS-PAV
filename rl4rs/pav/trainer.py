import json
import os

import numpy as np
import torch
import torch.nn as nn

from rl4rs.pav.dataset import add_returns, discrete_action_vector, ensure_dir, flatten_episodes
from rl4rs.pav.diagnostics import (
    alignment_with_logging_advantage,
    build_state_ids,
    distinguishability,
)
from rl4rs.pav.mc_estimator import estimate_q_mu, estimate_q_mu_trajectory_average, mc_metadata
from rl4rs.pav.models import RewardModel, Verifier, ZeroRewardModel, save_checkpoint
from rl4rs.pav.progress import (
    build_progress_pair_indices,
    combine_progress,
    compute_directional_progress,
    compute_k_step_progress,
    normalize_by_step,
    shape_rewards,
    verifier_labels,
)
from rl4rs.pav.prover import apply_prover_actions_to_flat, build_prover, prover_metadata


def _device(config):
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _sample_indices(total, max_samples):
    indices = np.arange(total)
    if max_samples is not None and total > max_samples:
        indices = np.random.choice(indices, size=max_samples, replace=False)
    return indices


def _batches(indices, batch_size, shuffle=True):
    indices = np.asarray(indices)
    if shuffle:
        indices = np.random.permutation(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def _tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(array, dtype=dtype, device=device)


def _reward_embed_dim(config):
    return int(config.embed_dim) if float(config.directional_lambda) > 0.0 else 0


def _value_targets(flat, config):
    target_kind = str(getattr(config, "reward_target", "q_mu"))
    if target_kind == "q_mu" and "q_mu" in flat:
        return np.asarray(flat["q_mu"], dtype=np.float32).reshape(-1)
    return np.asarray(flat["returns"], dtype=np.float32).reshape(-1)


def _slate_env_context(config):
    from rl4rs.online.config import build_online_slate_config
    from rl4rs.online.env_utils import attach_slate_masks, make_slate_env

    sim_batch = int(getattr(config, "mc_sim_batch_size", 1) or 1)
    # RecSimBase naming is inverted: gpu=True -> CPU sim, gpu=False -> TF uses GPU.
    sim_gpu_flag = not bool(getattr(config, "mc_sim_use_cpu", False))
    env_config = build_online_slate_config(
        config.output_dir,
        config.dataset_dir,
        batch_size=max(1, sim_batch),
        gpu=sim_gpu_flag,
    )
    attach_slate_masks(env_config)
    return env_config, make_slate_env


def train_reward_model(flat, config):
    observations = flat["observations"]
    targets = _value_targets(flat, config)
    device = _device(config)
    embed_dim = _reward_embed_dim(config)

    if config.reward_model_zero:
        return ZeroRewardModel(embed_dim=embed_dim), {
            "value_mse": float(np.mean(np.square(targets))),
            "embed_dim": embed_dim,
        }

    model = RewardModel(observations.shape[1], config.hidden_units, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    train_indices = _sample_indices(len(targets), config.max_train_samples)

    for epoch in range(config.reward_epochs):
        losses = []
        model.train()
        for batch_idx in _batches(train_indices, config.batch_size):
            x = _tensor(observations[batch_idx], device)
            y = _tensor(targets[batch_idx], device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        print("PAV reward epoch {} mse {:.6f}".format(epoch + 1, float(np.mean(losses))))

    values = predict_reward_values(model, observations, config)
    mse = float(np.mean(np.square(values - targets)))
    ensure_dir(config.reward_model_path)
    save_checkpoint(config.reward_model_path, model, {
        "observation_dim": int(observations.shape[1]),
        "hidden_units": config.hidden_units,
        "embed_dim": embed_dim,
        "value_mse": mse,
    })
    return model, {"value_mse": mse, "embed_dim": embed_dim}


def predict_reward_values(model, observations, config):
    device = _device(config)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx in _batches(np.arange(len(observations)), config.batch_size, shuffle=False):
            x = _tensor(observations[batch_idx], device)
            outputs.append(model(x).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype("float32")


def predict_reward_embeddings(model, observations, config):
    embed_dim = _reward_embed_dim(config)
    if embed_dim <= 0:
        return None
    device = _device(config)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx in _batches(np.arange(len(observations)), config.batch_size, shuffle=False):
            x = _tensor(observations[batch_idx], device)
            outputs.append(model.encode(x, normalize=True).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype("float32")


def finetune_reward_consistency(flat, model, verifier_scores, progress_pairs, config,
                                directional_progress=None):
    """Chopsticks-style consistency: penalize |progress| when verifier is low."""
    beta = float(config.consistency_beta)
    if beta <= 0.0 or config.reward_model_zero:
        return model, {"consistency_loss": None}

    observations = flat["observations"]
    device = _device(config)
    gamma = float(config.gamma)
    k = int(config.k)
    directional_lambda = float(config.directional_lambda)
    if directional_progress is not None:
        directional_progress = np.asarray(directional_progress, dtype=np.float32)

    t_indices = progress_pairs["t_indices"]
    tk_indices = progress_pairs["tk_indices"]
    reward_sums = progress_pairs["reward_sums"]
    bootstrap_masks = progress_pairs["bootstrap_masks"]
    verifier_scores = np.asarray(verifier_scores, dtype="float32")

    train_indices = _sample_indices(len(t_indices), config.max_train_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate * 0.5)
    value_criterion = nn.MSELoss()

    last_consistency = None
    model.train()
    for epoch in range(config.consistency_epochs):
        value_losses = []
        consistency_losses = []
        for batch_idx in _batches(train_indices, config.batch_size):
            pair_idx = batch_idx
            x_t = _tensor(observations[t_indices[pair_idx]], device)
            x_tk = _tensor(observations[tk_indices[pair_idx]], device)
            y = _tensor(flat["returns"][t_indices[pair_idx]], device)

            value_pred = model(x_t)
            value_loss = value_criterion(value_pred, y)

            r_t = model(x_t)
            r_tk = model(x_tk)
            reward_const = _tensor(reward_sums[pair_idx], device)
            bootstrap = _tensor(bootstrap_masks[pair_idx], device)
            progress = reward_const + bootstrap * (gamma ** k) * r_tk - r_t

            if directional_progress is not None and directional_lambda > 0.0:
                dir_const = _tensor(directional_progress[t_indices[pair_idx]], device)
                progress = progress + directional_lambda * dir_const

            v_gate = _tensor(verifier_scores[t_indices[pair_idx]], device)
            consistency_loss = torch.mean((1.0 - v_gate) * torch.abs(progress))

            loss = value_loss + beta * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value_losses.append(float(value_loss.detach().cpu().item()))
            consistency_losses.append(float(consistency_loss.detach().cpu().item()))

        last_consistency = float(np.mean(consistency_losses))
        print(
            "PAV consistency epoch {} value_mse {:.6f} consistency {:.6f}".format(
                epoch + 1, float(np.mean(value_losses)), last_consistency
            )
        )

    ensure_dir(config.reward_model_path)
    embed_dim = _reward_embed_dim(config)
    values = predict_reward_values(model, observations, config)
    mse = float(np.mean(np.square(values - flat["returns"])))
    save_checkpoint(config.reward_model_path, model, {
        "observation_dim": int(observations.shape[1]),
        "hidden_units": config.hidden_units,
        "embed_dim": embed_dim,
        "value_mse": mse,
        "consistency_beta": beta,
    })
    return model, {"consistency_loss": last_consistency, "value_mse": mse}


def _binary_auc(labels, scores):
    labels = np.asarray(labels).astype("int32")
    scores = np.asarray(scores)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = float(np.sum(ranks[pos]))
    auc = (pos_rank_sum - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    return float(auc)


def _contribution_return_correlation(contribution, returns):
    contribution = np.asarray(contribution, dtype="float64")
    returns = np.asarray(returns, dtype="float64")
    if contribution.std() < 1e-8 or returns.std() < 1e-8:
        return None
    return float(np.corrcoef(contribution, returns)[0, 1])


def train_verifier(flat, labels, config, q_targets=None):
    observations = flat["observations"]
    actions = discrete_action_vector(flat["actions"])
    actions = np.clip(actions, 0, config.action_size - 1)
    device = _device(config)
    output_mode = str(getattr(config, "verifier_output_mode", "binary"))

    model = Verifier(
        observations.shape[1],
        config.action_size,
        config.hidden_units,
        output_mode=output_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()
    train_indices = _sample_indices(len(labels), config.max_train_samples)
    use_q = output_mode in ("q_regression", "dual") and q_targets is not None
    q_targets = np.asarray(q_targets, dtype=np.float32) if use_q else None

    for epoch in range(config.verifier_epochs):
        losses = []
        model.train()
        for batch_idx in _batches(train_indices, config.batch_size):
            x = _tensor(observations[batch_idx], device)
            a = _tensor(actions[batch_idx], device, dtype=torch.long)
            y = _tensor(labels[batch_idx], device)
            optimizer.zero_grad()
            if output_mode == "q_regression":
                pred_q = model(x, a)
                target_q = _tensor(q_targets[batch_idx], device)
                loss = huber(pred_q, target_q)
            elif output_mode == "dual":
                gate_logits, pred_q = model(x, a)
                target_q = _tensor(q_targets[batch_idx], device)
                loss = bce(gate_logits, y) + huber(pred_q, target_q)
            else:
                logits = model(x, a)
                loss = bce(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        print("PAV verifier epoch {} loss {:.6f}".format(epoch + 1, float(np.mean(losses))))

    scores = predict_verifier_scores(model, observations, actions, config)
    predictions = (scores >= 0.5).astype("float32")
    accuracy = float(np.mean(predictions == labels))
    auc = _binary_auc(labels, scores)
    q_mse = None
    if use_q:
        pred_q = predict_verifier_q(model, observations, actions, config)
        q_mse = float(np.mean(np.square(pred_q - q_targets)))
    ensure_dir(config.verifier_path)
    save_checkpoint(config.verifier_path, model, {
        "observation_dim": int(observations.shape[1]),
        "action_size": int(config.action_size),
        "hidden_units": config.hidden_units,
        "accuracy": accuracy,
        "auc": auc,
        "q_mse": q_mse,
        "verifier_label_mode": config.verifier_label_mode,
        "verifier_output_mode": output_mode,
    })
    metrics = {"verifier_accuracy": accuracy, "verifier_auc": auc, "verifier_q_mse": q_mse}
    return model, metrics


def predict_verifier_q(model, observations, actions, config):
    device = _device(config)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx in _batches(np.arange(len(observations)), config.batch_size, shuffle=False):
            x = _tensor(observations[batch_idx], device)
            a = _tensor(actions[batch_idx], device, dtype=torch.long)
            if hasattr(model, "predict_q"):
                outputs.append(model.predict_q(x, a).detach().cpu().numpy())
            else:
                outputs.append(model(x, a).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype("float32")


def predict_verifier_scores(model, observations, actions, config):
    device = _device(config)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx in _batches(np.arange(len(observations)), config.batch_size, shuffle=False):
            x = _tensor(observations[batch_idx], device)
            a = _tensor(actions[batch_idx], device, dtype=torch.long)
            if getattr(model, "output_mode", "binary") == "dual":
                gate_logits, _pred_q = model(x, a)
                logits = gate_logits
            elif getattr(model, "output_mode", "binary") == "q_regression":
                logits = model.predict_q(x, a)
                outputs.append(logits.detach().cpu().numpy())
                continue
            else:
                logits = model(x, a)
            outputs.append(torch.sigmoid(logits).detach().cpu().numpy())
    if getattr(model, "output_mode", "binary") == "q_regression":
        q = np.concatenate(outputs, axis=0).astype("float32")
        return np.clip(q / (np.std(q) + 1e-6), 0.0, 1.0)
    return np.concatenate(outputs, axis=0).astype("float32")


def _compute_contribution(flat, progress, values, verifier, verifier_scores, config):
    output_mode = str(getattr(config, "verifier_output_mode", "binary"))
    if config.use_raw_progress:
        return np.asarray(progress, dtype="float32")
    if not config.use_verifier or verifier is None:
        return np.asarray(progress, dtype="float32")
    actions = np.clip(discrete_action_vector(flat["actions"]), 0, config.action_size - 1)
    if output_mode in ("q_regression", "dual"):
        q_sa = predict_verifier_q(verifier, flat["observations"], actions, config)
        advantage = q_sa - np.asarray(values, dtype="float32")
        if output_mode == "dual":
            return advantage * np.asarray(verifier_scores, dtype="float32")
        return advantage.astype("float32")
    return (np.asarray(progress, dtype="float32") * np.asarray(verifier_scores, dtype="float32"))


def _simulator_available(config):
    model_path = os.path.join(config.output_dir, "simulator_a_dien", "model")
    return (
        os.path.isfile(model_path)
        or os.path.isfile(model_path + ".index")
        or os.path.isfile(model_path + ".meta")
    )


def build_pav_signals(dataset, config):
    from rl4rs.pav.suite_progress import note as _suite_note

    flat = flatten_episodes(dataset)
    flat = add_returns(flat, config.gamma)
    _suite_note("flatten_returns")
    env_config, env_factory = _slate_env_context(config)
    device = _device(config)
    prover, _prover_rng = build_prover(
        config, flat=flat, env_config=env_config, device=device
    )
    flat = apply_prover_actions_to_flat(flat, prover, env_config, config, _prover_rng)
    _suite_note("prover_ready", actions_source=flat.get("actions_source"))
    print(
        "PAV actions_source={} (logged actions preserved in flat['logged_actions'])".format(
            flat.get("actions_source", "unknown")
        ),
        flush=True,
    )

    if getattr(config, "use_trajectory_q_avg", True):
        flat["q_mu"] = estimate_q_mu_trajectory_average(flat, gamma=config.gamma)
    else:
        flat["q_mu"] = np.asarray(flat["returns"], dtype=np.float32).copy()

    reward_model, reward_metrics = train_reward_model(flat, config)
    values = predict_reward_values(reward_model, flat["observations"], config)
    _suite_note("reward_v1", value_mse=reward_metrics.get("value_mse"))

    if getattr(config, "use_hybrid_mc", False) or getattr(config, "use_simulator_q", False):
        if _simulator_available(config):
            try:
                _suite_note("hybrid_mc")
                flat["q_mu"] = estimate_q_mu(
                    flat,
                    config,
                    prover=prover,
                    env_config=env_config,
                    env_factory=env_factory,
                    value_baseline=values,
                )
                reward_model, reward_metrics = train_reward_model(flat, config)
                values = predict_reward_values(reward_model, flat["observations"], config)
                _suite_note("reward_v2_mc", value_mse=reward_metrics.get("value_mse"))
            except Exception as exc:
                print("PAV hybrid/simulator MC skipped: {}".format(exc), flush=True)
                _suite_note("hybrid_mc_skipped", error=str(exc))
        else:
            print("PAV: simulator checkpoint missing, skipping hybrid/full MC", flush=True)
            _suite_note("hybrid_mc_skipped", error="simulator_missing")

    potential_progress = compute_k_step_progress(flat, values, config.k, config.gamma)
    directional_progress = None
    if float(config.directional_lambda) > 0.0:
        embeddings = predict_reward_embeddings(reward_model, flat["observations"], config)
        directional_progress = compute_directional_progress(flat, embeddings, config.k)
        progress = combine_progress(
            potential_progress, directional_progress, config.directional_lambda
        )
    else:
        progress = potential_progress

    labels, progress_baseline, return_baseline = verifier_labels(
        progress,
        flat["returns"],
        flat["step_ids"],
        mode=config.verifier_label_mode,
        margin_frac=config.verifier_margin_frac,
        state_values=values,
    )

    verifier_scores = np.ones_like(progress, dtype="float32")
    verifier_metrics = {"verifier_accuracy": None, "verifier_auc": None, "verifier_q_mse": None}
    verifier = None
    if config.use_verifier:
        q_targets = flat.get("q_mu", flat["returns"])
        verifier, verifier_metrics = train_verifier(flat, labels, config, q_targets=q_targets)
        actions = np.clip(discrete_action_vector(flat["actions"]), 0, config.action_size - 1)
        verifier_scores = predict_verifier_scores(
            verifier, flat["observations"], actions, config
        )
        _suite_note(
            "verifier",
            verifier_q_mse=verifier_metrics.get("verifier_q_mse"),
            verifier_auc=verifier_metrics.get("verifier_auc"),
        )

    consistency_metrics = {"consistency_loss": None}
    if float(config.consistency_beta) > 0.0 and config.use_verifier:
        progress_pairs = build_progress_pair_indices(flat, config.k)
        reward_model, consistency_metrics = finetune_reward_consistency(
            flat,
            reward_model,
            verifier_scores,
            progress_pairs,
            config,
            directional_progress=directional_progress,
        )
        values = predict_reward_values(reward_model, flat["observations"], config)
        potential_progress = compute_k_step_progress(flat, values, config.k, config.gamma)
        if float(config.directional_lambda) > 0.0:
            embeddings = predict_reward_embeddings(reward_model, flat["observations"], config)
            directional_progress = compute_directional_progress(flat, embeddings, config.k)
            progress = combine_progress(
                potential_progress, directional_progress, config.directional_lambda
            )
        else:
            progress = potential_progress
        reward_metrics.update(consistency_metrics)

    contribution = _compute_contribution(
        flat, progress, values, verifier, verifier_scores, config
    )
    shaped_rewards, normalized_contribution, norm_stats = shape_rewards(
        flat["rewards"],
        contribution,
        flat["step_ids"],
        alpha=config.alpha,
        clip_c=config.clip_c,
        use_clipping=config.use_clipping,
        normalize_contribution=getattr(config, "normalize_contribution", False),
    )
    _, progress_norm_stats = normalize_by_step(progress, flat["step_ids"])

    state_ids = build_state_ids(flat)
    dist_overall, dist_per_step = distinguishability(contribution, state_ids)
    align_overall, align_per_step = alignment_with_logging_advantage(
        contribution, flat["returns"], values, flat["step_ids"]
    )

    stats = {
        "env": config.env,
        "trial_name": config.trial_name,
        "k": config.k,
        "alpha": config.alpha,
        "clip_c": config.clip_c,
        "use_verifier": config.use_verifier,
        "use_raw_progress": config.use_raw_progress,
        "use_clipping": config.use_clipping,
        "normalize_contribution": bool(getattr(config, "normalize_contribution", False)),
        "reward_target": str(getattr(config, "reward_target", "q_mu")),
        "directional_lambda": float(config.directional_lambda),
        "embed_dim": _reward_embed_dim(config),
        "verifier_label_mode": config.verifier_label_mode,
        "verifier_output_mode": str(getattr(config, "verifier_output_mode", "q_regression")),
        "consistency_beta": float(config.consistency_beta),
        "distinguishability_floor": float(getattr(config, "distinguishability_floor", 0.05)),
        "alpha_decay_enabled": bool(getattr(config, "alpha_decay_enabled", True)),
        "alpha_decay_rate": float(getattr(config, "alpha_decay_rate", 0.1)),
        "reward_metrics": reward_metrics,
        "verifier_metrics": verifier_metrics,
        "consistency_metrics": consistency_metrics,
        "potential_progress_mean": float(np.mean(potential_progress)),
        "potential_progress_std": float(np.std(potential_progress)),
        "directional_progress_mean": (
            float(np.mean(directional_progress)) if directional_progress is not None else None
        ),
        "directional_progress_std": (
            float(np.std(directional_progress)) if directional_progress is not None else None
        ),
        "progress_mean": float(np.mean(progress)),
        "progress_std": float(np.std(progress)),
        "contribution_mean": float(np.mean(contribution)),
        "contribution_std": float(np.std(contribution)),
        "contribution_return_corr": _contribution_return_correlation(contribution, flat["returns"]),
        "z_positive_rate": float(np.mean(labels)),
        "norm_by_step": norm_stats,
        "progress_norm_by_step": progress_norm_stats,
        "distinguishability": dist_overall,
        "distinguishability_per_step": dist_per_step,
        "alignment_corr": align_overall,
        "alignment_corr_per_step": align_per_step,
        "actions_source": flat.get("actions_source"),
    }
    stats.update(prover_metadata(config, prover))
    stats.update(mc_metadata(config))
    ensure_dir(config.stats_path)
    with open(config.stats_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    _suite_note(
        "done",
        status="complete",
        value_mse=(reward_metrics or {}).get("value_mse"),
        contribution_return_corr=stats.get("contribution_return_corr"),
    )

    signals = {
        "flat": flat,
        "values": values,
        "potential_progress": potential_progress,
        "directional_progress": directional_progress,
        "progress": progress,
        "labels": labels,
        "progress_baseline": progress_baseline,
        "return_baseline": return_baseline,
        "verifier_scores": verifier_scores,
        "contribution": contribution,
        "normalized_contribution": normalized_contribution,
        "shaped_rewards": shaped_rewards,
        "stats": stats,
    }
    return signals


def fit_pav_models(dataset, config):
    """Train PAV reward/verifier checkpoints. Alias for build_pav_signals."""
    return build_pav_signals(dataset, config)
