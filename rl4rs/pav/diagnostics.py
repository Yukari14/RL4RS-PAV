import csv
import os

import numpy as np


def ensure_dir(path):
    directory = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def build_state_ids(flat):
    """State buckets: (step_id, user-context segment) for conditional Var_a."""
    step_ids = np.asarray(flat["step_ids"], dtype=np.int64).reshape(-1)
    observations = np.asarray(flat["observations"], dtype=np.float32)
    if observations.ndim == 1:
        observations = observations.reshape(1, -1)
    user_seg = np.sum((observations[:, :8] * 100.0).astype(np.int64), axis=1) % 10007
    episode_ids = np.asarray(flat.get("episode_ids", np.zeros_like(step_ids)), dtype=np.int64)
    return step_ids * 1000003 + user_seg * 997 + (episode_ids % 997)


def online_rolling_distinguishability(contributions, step_ids, min_count=2):
    """Online proxy: Var[contribution | step_id] pooled across parallel envs.

    Offline stats use build_state_ids (step × user) with multiple actions per state
    from logged/prover data. Online RL sees one action per (user, step), so we bucket
    by step_id only; with batch_size>1 each step accumulates many user samples.
    """
    return distinguishability(contributions, step_ids, min_count=min_count)


def distinguishability(values, state_ids, min_count=2):
    """E_s Var_{a~data}[values(s,a)] with per-bucket variance."""
    values = np.asarray(values, dtype=np.float64)
    state_ids = np.asarray(state_ids, dtype=np.int64).reshape(-1)
    per_bucket = {}
    for bucket in np.unique(state_ids):
        idx = state_ids == bucket
        count = int(np.sum(idx))
        if count < min_count:
            continue
        per_bucket[str(int(bucket))] = float(np.var(values[idx]))
    if not per_bucket:
        return None, {}
    overall = float(np.mean(list(per_bucket.values())))
    return overall, per_bucket


def alignment_with_logging_advantage(progress, returns, state_values, step_ids):
    """Step-stratified correlation between A^logging and A^mu proxy."""
    progress = np.asarray(progress, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    state_values = np.asarray(state_values, dtype=np.float64)
    step_ids = np.asarray(step_ids, dtype=np.int64).reshape(-1)
    a_logging = returns - state_values
    a_mu = progress
    per_step = {}
    for step_id in np.unique(step_ids):
        idx = step_ids == step_id
        if np.sum(idx) < 3:
            continue
        x = a_logging[idx]
        y = a_mu[idx]
        if x.std() < 1e-8 or y.std() < 1e-8:
            continue
        per_step[str(int(step_id))] = float(np.corrcoef(x, y)[0, 1])
    if not per_step:
        return None, {}
    overall = float(np.mean(list(per_step.values())))
    return overall, per_step


def format_prover_quality(stats):
    lines = ["[Prover quality]"]
    dist = stats.get("distinguishability")
    align = stats.get("alignment_corr")
    lines.append(
        "  distinguishability  = {}   (paper recommends > 0.05)".format(
            "{:.4f}".format(dist) if dist is not None else "n/a"
        )
    )
    lines.append(
        "  alignment_corr      = {}   (should be > 0; negative => bad prover)".format(
            "{:.4f}".format(align) if align is not None else "n/a"
        )
    )
    return lines


def write_step_diagnostics(signals, output_path):
    flat = signals["flat"]
    ensure_dir(output_path)
    step_ids = flat["step_ids"]
    fields = [
        "step",
        "count",
        "original_reward_mean",
        "progress_mean",
        "verifier_score_mean",
        "contribution_mean",
        "shaped_reward_mean",
        "z_positive_rate",
        "distinguishability",
        "alignment_corr",
    ]
    state_ids = build_state_ids(flat)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step_id in np.unique(step_ids):
            idx = step_ids == step_id
            dist_step, _ = distinguishability(signals["contribution"][idx], state_ids[idx])
            align_step, _ = alignment_with_logging_advantage(
                signals["contribution"][idx],
                flat["returns"][idx],
                signals["values"][idx],
                step_ids[idx],
            )
            writer.writerow({
                "step": int(step_id),
                "count": int(np.sum(idx)),
                "original_reward_mean": float(np.mean(flat["rewards"][idx])),
                "progress_mean": float(np.mean(signals["progress"][idx])),
                "verifier_score_mean": float(np.mean(signals["verifier_scores"][idx])),
                "contribution_mean": float(np.mean(signals["contribution"][idx])),
                "shaped_reward_mean": float(np.mean(signals["shaped_rewards"][idx])),
                "z_positive_rate": float(np.mean(signals["labels"][idx])),
                "distinguishability": dist_step,
                "alignment_corr": align_step,
            })
    return output_path


def collect_scatter_rows(stats_dir, ablation_csv=None, rl_summary_glob=None):
    """Collect (label, distinguishability, return_proxy) from stats JSON / ablation / RL summaries."""
    import glob
    import json

    rows = []
    if stats_dir and os.path.isdir(stats_dir):
        for path in sorted(glob.glob(os.path.join(stats_dir, "stats_*.json"))):
            with open(path) as f:
                stats = json.load(f)
            label = os.path.basename(path).replace("stats_", "").replace(".json", "")
            rows.append({
                "label": label,
                "distinguishability": stats.get("distinguishability"),
                "return_proxy": stats.get("contribution_return_corr"),
                "alignment_corr": stats.get("alignment_corr"),
                "source": "stats",
            })

    if ablation_csv and os.path.isfile(ablation_csv):
        with open(ablation_csv, newline="") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "label": row.get("variant") or row.get("suffix"),
                    "distinguishability": _float_or_none(row.get("distinguishability")),
                    "return_proxy": _float_or_none(row.get("contribution_return_corr")),
                    "alignment_corr": _float_or_none(row.get("alignment_corr")),
                    "source": "ablation",
                })

    if rl_summary_glob:
        for path in sorted(glob.glob(rl_summary_glob)):
            with open(path) as f:
                summary = json.load(f)
            eval_raw = summary.get("eval_raw_sim") or {}
            rows.append({
                "label": summary.get("condition", os.path.basename(path)),
                "distinguishability": None,
                "return_proxy": eval_raw.get("sim_avg_reward"),
                "alignment_corr": None,
                "source": "rl",
            })
    return rows


def _float_or_none(value):
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_distinguishability_return_scatter(rows, output_path, title="Distinguishability vs Return Proxy"):
    """Scatter plot for §4(d): distinguishability × final_return (or corr proxy offline)."""
    import matplotlib.pyplot as plt

    pts = [
        r for r in rows
        if r.get("distinguishability") is not None and r.get("return_proxy") is not None
    ]
    if not pts:
        raise ValueError("No rows with both distinguishability and return_proxy.")

    ensure_dir(output_path)
    xs = [r["distinguishability"] for r in pts]
    ys = [r["return_proxy"] for r in pts]
    labels = [str(r.get("label", "")) for r in pts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=60, alpha=0.75, edgecolors="k", linewidths=0.5)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=1, label="dist floor 0.05")
    ax.set_xlabel("Distinguishability  E_s Var_a[A^mu]")
    ax.set_ylabel("Return proxy (corr offline / sim_eval online)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_experiment_matrix(output_path):
    ensure_dir(output_path)
    rows = []
    for env in ["SlateRecEnv-v0", "SeqSlateRecEnv-v0"]:
        for algo in ["CQL", "BCQ", "BC"]:
            rows.append({
                "environment": env,
                "algorithm": algo,
                "pav": "No",
                "command": "python -u batchrl_train.py {} train \"{{'env':'{}'}}\"".format(algo, env),
            })
            rows.append({
                "environment": env,
                "algorithm": algo,
                "pav": "Yes",
                "command": "python -u batchrl_train.py {} train \"{{'env':'{}','use_pav':True}}\"".format(algo, env),
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["environment", "algorithm", "pav", "command"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_ablation_matrix(output_path):
    ensure_dir(output_path)
    rows = []
    for k in [1, 3, 5]:
        rows.append({"ablation": "k", "k": k, "alpha": 0.1, "reward_model_zero": False,
                     "use_verifier": True, "use_raw_progress": False, "use_clipping": True})
    for alpha in [0.05, 0.1, 0.2]:
        rows.append({"ablation": "alpha", "k": 3, "alpha": alpha, "reward_model_zero": False,
                     "use_verifier": True, "use_raw_progress": False, "use_clipping": True})
    rows.extend([
        {"ablation": "reward_model_zero", "k": 3, "alpha": 0.1, "reward_model_zero": True,
         "use_verifier": True, "use_raw_progress": False, "use_clipping": True},
        {"ablation": "raw_progress", "k": 3, "alpha": 0.1, "reward_model_zero": False,
         "use_verifier": False, "use_raw_progress": True, "use_clipping": True},
        {"ablation": "no_verifier_gate", "k": 3, "alpha": 0.1, "reward_model_zero": False,
         "use_verifier": False, "use_raw_progress": False, "use_clipping": True},
        {"ablation": "without_clipping", "k": 3, "alpha": 0.1, "reward_model_zero": False,
         "use_verifier": True, "use_raw_progress": False, "use_clipping": False},
    ])
    for capacity in ["small", "medium", "large"]:
        rows.append({"ablation": "capacity_{}".format(capacity), "k": 3, "alpha": 0.1,
                     "reward_model_zero": False, "use_verifier": True,
                     "use_raw_progress": False, "use_clipping": True})

    v2_rows = [
        {"ablation": "v2_directional", "directional_lambda": 0.5,
         "verifier_label_mode": "sign", "consistency_beta": 0.0,
         "verifier_output_mode": "q_regression", "normalize_contribution": False},
        {"ablation": "v2_necessity", "directional_lambda": 0.0,
         "verifier_label_mode": "necessity", "consistency_beta": 0.0,
         "verifier_output_mode": "q_regression", "normalize_contribution": False},
        {"ablation": "v2_full", "directional_lambda": 0.5,
         "verifier_label_mode": "necessity_combined", "consistency_beta": 0.1,
         "verifier_output_mode": "q_regression", "normalize_contribution": False},
        {"ablation": "v2_binary_baseline", "directional_lambda": 0.5,
         "verifier_label_mode": "necessity_combined", "consistency_beta": 0.1,
         "verifier_output_mode": "binary", "normalize_contribution": True},
    ]
    for row in v2_rows:
        row.setdefault("k", 3)
        row.setdefault("alpha", 0.1)
        row.setdefault("reward_model_zero", False)
        row.setdefault("use_verifier", True)
        row.setdefault("use_raw_progress", False)
        row.setdefault("use_clipping", True)
        rows.append(row)

    fieldnames = ["ablation", "k", "alpha", "reward_model_zero", "use_verifier",
                  "use_raw_progress", "use_clipping", "directional_lambda",
                  "verifier_label_mode", "consistency_beta", "verifier_output_mode",
                  "normalize_contribution", "use_trajectory_q_avg", "use_hybrid_mc",
                  "track_distinguishability", "track_alignment_corr"]
    for row in rows:
        row.setdefault("directional_lambda", 0.0)
        row.setdefault("verifier_label_mode", "sign")
        row.setdefault("consistency_beta", 0.0)
        row.setdefault("verifier_output_mode", "q_regression")
        row.setdefault("normalize_contribution", False)
        row.setdefault("use_trajectory_q_avg", True)
        row.setdefault("use_hybrid_mc", True)
        row.setdefault("track_distinguishability", True)
        row.setdefault("track_alignment_corr", True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
