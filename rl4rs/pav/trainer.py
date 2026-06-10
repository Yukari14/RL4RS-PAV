import json
import os

import numpy as np
import torch
import torch.nn as nn

from rl4rs.pav.dataset import add_returns, discrete_action_vector, ensure_dir, flatten_episodes
from rl4rs.pav.models import RewardModel, Verifier, ZeroRewardModel, save_checkpoint
from rl4rs.pav.progress import (
    build_progress_pair_indices,
    combine_progress,
    compute_directional_progress,
    compute_k_step_progress,
    shape_rewards,
    verifier_labels,
)


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


def train_reward_model(flat, config):
    observations = flat["observations"]
    targets = flat["returns"]
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


def train_verifier(flat, labels, config):
    observations = flat["observations"]
    actions = discrete_action_vector(flat["actions"])
    actions = np.clip(actions, 0, config.action_size - 1)
    device = _device(config)

    model = Verifier(observations.shape[1], config.action_size, config.hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train_indices = _sample_indices(len(labels), config.max_train_samples)

    for epoch in range(config.verifier_epochs):
        losses = []
        model.train()
        for batch_idx in _batches(train_indices, config.batch_size):
            x = _tensor(observations[batch_idx], device)
            a = _tensor(actions[batch_idx], device, dtype=torch.long)
            y = _tensor(labels[batch_idx], device)
            logits = model(x, a)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        print("PAV verifier epoch {} bce {:.6f}".format(epoch + 1, float(np.mean(losses))))

    scores = predict_verifier_scores(model, observations, actions, config)
    predictions = (scores >= 0.5).astype("float32")
    accuracy = float(np.mean(predictions == labels))
    auc = _binary_auc(labels, scores)
    ensure_dir(config.verifier_path)
    save_checkpoint(config.verifier_path, model, {
        "observation_dim": int(observations.shape[1]),
        "action_size": int(config.action_size),
        "hidden_units": config.hidden_units,
        "accuracy": accuracy,
        "auc": auc,
        "verifier_label_mode": config.verifier_label_mode,
    })
    return model, {"verifier_accuracy": accuracy, "verifier_auc": auc}


def predict_verifier_scores(model, observations, actions, config):
    device = _device(config)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx in _batches(np.arange(len(observations)), config.batch_size, shuffle=False):
            x = _tensor(observations[batch_idx], device)
            a = _tensor(actions[batch_idx], device, dtype=torch.long)
            logits = model(x, a)
            outputs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype("float32")


def build_pav_signals(dataset, config):
    flat = flatten_episodes(dataset)
    flat = add_returns(flat, config.gamma)
    reward_model, reward_metrics = train_reward_model(flat, config)
    values = predict_reward_values(reward_model, flat["observations"], config)

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
    verifier_metrics = {"verifier_accuracy": None, "verifier_auc": None}
    if config.use_verifier:
        verifier, verifier_metrics = train_verifier(flat, labels, config)
        actions = np.clip(discrete_action_vector(flat["actions"]), 0, config.action_size - 1)
        verifier_scores = predict_verifier_scores(
            verifier, flat["observations"], actions, config
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

    contribution = progress if config.use_raw_progress else progress * verifier_scores
    shaped_rewards, normalized_contribution, norm_stats = shape_rewards(
        flat["rewards"],
        contribution,
        flat["step_ids"],
        alpha=config.alpha,
        clip_c=config.clip_c,
        use_clipping=config.use_clipping,
    )
    _, progress_norm_stats = normalize_by_step(progress, flat["step_ids"])

    stats = {
        "env": config.env,
        "trial_name": config.trial_name,
        "k": config.k,
        "alpha": config.alpha,
        "clip_c": config.clip_c,
        "use_verifier": config.use_verifier,
        "use_raw_progress": config.use_raw_progress,
        "use_clipping": config.use_clipping,
        "directional_lambda": float(config.directional_lambda),
        "embed_dim": _reward_embed_dim(config),
        "verifier_label_mode": config.verifier_label_mode,
        "consistency_beta": float(config.consistency_beta),
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
    }
    ensure_dir(config.stats_path)
    with open(config.stats_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

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
