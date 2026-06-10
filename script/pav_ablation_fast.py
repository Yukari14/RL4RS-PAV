#!/usr/bin/env python3
"""Fast PAV v2 ablation: signal-level diagnostics only (no 500-epoch RL).

Runs build_pav_signals on a dataset subset and compares variants in minutes.
Use contribution_return_corr + verifier_auc as gates before any short RL pilot.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from rl4rs.pav.config import PAVConfig
from rl4rs.pav.dataset import load_mdpdataset
from rl4rs.pav.trainer import build_pav_signals


def _subset_dataset(dataset, max_episodes):
    if max_episodes is None or max_episodes <= 0:
        return dataset
    episodes = dataset.episodes[:max_episodes]
    observations = np.concatenate([ep.observations for ep in episodes], axis=0)
    actions = np.concatenate([ep.actions for ep in episodes], axis=0)
    rewards = np.concatenate([ep.rewards for ep in episodes], axis=0)
    terminals = []
    for ep in episodes:
        t = getattr(ep, "terminals", None)
        if t is None:
            t = np.zeros(len(ep.rewards), dtype="float32")
            if len(t):
                t[-1] = 1.0
        terminals.append(np.asarray(t, dtype="float32").reshape(-1)[: len(ep.rewards)])
    terminals = np.concatenate(terminals, axis=0)
    from d3rlpy.dataset import MDPDataset

    return MDPDataset(observations, actions, rewards, terminals)


def _variant_grid():
    """Minimal ablation ladder: each row adds one v2 component."""
    return [
        {
            "name": "baseline_sign",
            "suffix": "abl_baseline",
            "verifier_label_mode": "sign",
            "directional_lambda": 0.0,
            "consistency_beta": 0.0,
        },
        {
            "name": "directional",
            "suffix": "abl_directional",
            "verifier_label_mode": "sign",
            "directional_lambda": 0.5,
            "consistency_beta": 0.0,
        },
        {
            "name": "necessity",
            "suffix": "abl_necessity",
            "verifier_label_mode": "necessity",
            "directional_lambda": 0.0,
            "consistency_beta": 0.0,
        },
        {
            "name": "directional_necessity",
            "suffix": "abl_dir_nec",
            "verifier_label_mode": "necessity_combined",
            "directional_lambda": 0.5,
            "consistency_beta": 0.0,
        },
        {
            "name": "full_v2",
            "suffix": "abl_full_v2",
            "verifier_label_mode": "necessity_combined",
            "directional_lambda": 0.5,
            "consistency_beta": 0.1,
        },
        {
            "name": "full_v2_no_verifier",
            "suffix": "abl_full_raw",
            "verifier_label_mode": "necessity_combined",
            "directional_lambda": 0.5,
            "consistency_beta": 0.0,
            "use_verifier": False,
            "use_raw_progress": True,
        },
    ]


def _gate_pass(stats, auc_min=0.58, corr_min=0.05):
    verifier_auc = stats.get("verifier_metrics", {}).get("verifier_auc")
    corr = stats.get("contribution_return_corr")
    auc_ok = verifier_auc is None or verifier_auc >= auc_min
    corr_ok = corr is None or corr >= corr_min
    return auc_ok and corr_ok


def run_ablation(args):
    base = {
        "env": args.env,
        "trial_name": args.trial_name,
        "action_size": args.action_size,
        "gamma": 1.0,
        "alpha": args.alpha,
        "k": args.k,
        "batch_size": args.batch_size,
        "reward_epochs": args.reward_epochs,
        "verifier_epochs": args.verifier_epochs,
        "max_train_samples": args.max_train_samples,
        "output_dir": args.output_dir,
        "dataset_dir": args.dataset_dir,
    }
    dataset_path = os.path.join(args.dataset_dir, "{}_{}.h5".format(args.env, args.trial_name))
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError("Dataset not found: {}".format(dataset_path))

    dataset = load_mdpdataset(dataset_path)
    dataset = _subset_dataset(dataset, args.max_episodes)
    print("Loaded {} episodes for fast ablation".format(len(dataset.episodes)))

    rows = []
    variants = _variant_grid()
    if args.variants:
        names = set(args.variants.split(","))
        variants = [v for v in variants if v["name"] in names]

    for variant in variants:
        cfg_dict = dict(base)
        cfg_dict.update({k: v for k, v in variant.items() if k != "name"})
        config = PAVConfig.from_dict(cfg_dict)
        t0 = time.time()
        print("\n=== Variant: {} ===".format(variant["name"]))
        signals = build_pav_signals(dataset, config)
        elapsed = time.time() - t0
        stats = signals["stats"]
        row = {
            "variant": variant["name"],
            "suffix": config.suffix,
            "seconds": round(elapsed, 1),
            "value_mse": stats["reward_metrics"].get("value_mse"),
            "verifier_auc": stats["verifier_metrics"].get("verifier_auc"),
            "verifier_accuracy": stats["verifier_metrics"].get("verifier_accuracy"),
            "z_positive_rate": stats.get("z_positive_rate"),
            "progress_mean": stats.get("progress_mean"),
            "progress_std": stats.get("progress_std"),
            "contribution_mean": stats.get("contribution_mean"),
            "contribution_std": stats.get("contribution_std"),
            "contribution_return_corr": stats.get("contribution_return_corr"),
            "directional_lambda": stats.get("directional_lambda"),
            "verifier_label_mode": stats.get("verifier_label_mode"),
            "consistency_beta": stats.get("consistency_beta"),
            "consistency_loss": stats.get("consistency_metrics", {}).get("consistency_loss"),
            "gate_pass": _gate_pass(stats, args.auc_min, args.corr_min),
        }
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    ensure_dir = os.path.join(args.output_dir, "pav", "ablation")
    os.makedirs(ensure_dir, exist_ok=True)
    out_csv = os.path.join(ensure_dir, "fast_ablation_{}_{}.csv".format(
        args.env.replace("-", "_"), args.trial_name
    ))
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = [r["variant"] for r in rows if r["gate_pass"]]
    print("\nFast ablation CSV: {}".format(out_csv))
    print("Gate pass (auc>={}, corr>={}): {}".format(args.auc_min, args.corr_min, passed))
    return out_csv, rows


def main():
    parser = argparse.ArgumentParser(description="Fast PAV v2 signal ablation")
    parser.add_argument("--env", default="SlateRecEnv-v0")
    parser.add_argument("--trial-name", default="a_50k_logged")
    parser.add_argument("--dataset-dir", default=os.environ.get("rl4rs_dataset_dir", "../dataset"))
    parser.add_argument("--output-dir", default=os.environ.get("rl4rs_output_dir", "../output"))
    parser.add_argument("--action-size", type=int, default=284)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int, default=2000,
                        help="Subset size; 2000 eps ~ few min on GPU")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--reward-epochs", type=int, default=3)
    parser.add_argument("--verifier-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--auc-min", type=float, default=0.58)
    parser.add_argument("--corr-min", type=float, default=0.05)
    parser.add_argument("--variants", default=None,
                        help="Comma-separated subset, e.g. baseline_sign,directional,full_v2")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
