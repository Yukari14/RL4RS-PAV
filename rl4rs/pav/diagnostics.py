import csv
import os

import numpy as np


def ensure_dir(path):
    directory = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


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
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step_id in np.unique(step_ids):
            idx = step_ids == step_id
            writer.writerow({
                "step": int(step_id),
                "count": int(np.sum(idx)),
                "original_reward_mean": float(np.mean(flat["rewards"][idx])),
                "progress_mean": float(np.mean(signals["progress"][idx])),
                "verifier_score_mean": float(np.mean(signals["verifier_scores"][idx])),
                "contribution_mean": float(np.mean(signals["contribution"][idx])),
                "shaped_reward_mean": float(np.mean(signals["shaped_rewards"][idx])),
                "z_positive_rate": float(np.mean(signals["labels"][idx])),
            })
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

    fieldnames = ["ablation", "k", "alpha", "reward_model_zero", "use_verifier",
                  "use_raw_progress", "use_clipping"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
