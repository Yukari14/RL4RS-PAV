#!/usr/bin/env python3
"""Phase 0: SlateRecEnv local smoke for online RL + PAV ladder."""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from rl4rs.online.config import build_online_slate_config
from rl4rs.online.device import configure_runtime, simulator_config_gpu
from rl4rs.online.env_utils import make_slate_env, sample_masked_actions, mask_stats

def run_policy(env, policy, num_episodes, seed, config):
    rng = np.random.RandomState(seed)
    env.seed(seed)
    max_steps = config["max_steps"]
    batch_size = config["batch_size"]

    episode_returns = []
    terminal_rewards = []
    mask_trace = []

    episodes_done = 0
    while episodes_done < num_episodes:
        env.reset(reset_file=(episodes_done == 0))
        ep_return = np.zeros(batch_size, dtype=np.float64)
        last_step_reward = np.zeros(batch_size, dtype=np.float64)

        for step in range(max_steps):
            obs = env.state
            mask_trace.append(mask_stats(obs, config))

            if policy == "logged":
                action = env.offline_action
            elif policy == "random_masked":
                action = sample_masked_actions(obs, rng, config)
            else:
                raise ValueError("unknown policy: {}".format(policy))

            _next_obs, reward, done, _info = env.step(action)
            reward_arr = np.asarray(reward, dtype=np.float64)
            if reward_arr.ndim == 0:
                reward_arr = np.array([reward_arr])
            ep_return += reward_arr
            if step == max_steps - 1:
                last_step_reward = reward_arr.copy()

        episode_returns.extend(ep_return.tolist())
        terminal_rewards.extend(last_step_reward.tolist())
        episodes_done += batch_size

    episode_returns = np.asarray(episode_returns[:num_episodes], dtype=np.float64)
    terminal_rewards = np.asarray(terminal_rewards[:num_episodes], dtype=np.float64)

    return {
        "policy": policy,
        "n_episodes": int(num_episodes),
        "seed": int(seed),
        "return_mean": float(episode_returns.mean()),
        "return_std": float(episode_returns.std()),
        "return_min": float(episode_returns.min()),
        "return_max": float(episode_returns.max()),
        "terminal_mean": float(terminal_rewards.mean()),
        "terminal_std": float(terminal_rewards.std()),
        "terminal_nonzero_rate": float(np.mean(terminal_rewards > 0)),
        "mask_final_mean_valid": float(mask_trace[-1]["mean_valid"]) if mask_trace else None,
        "mask_shrink_steps": int(sum(
            1 for i in range(1, len(mask_trace))
            if mask_trace[i]["mean_valid"] < mask_trace[i - 1]["mean_valid"]
        )),
    }


def compute_gate_thresholds(logged_mean):
    return {"abs_sim_gate": max(3.0, 0.05 * logged_mean), "rel_sim_gate": 0.05}


def evaluate_phase0_gate(logged, random_policy, config):
    checks = []
    model_path = config["model_file"]
    checks.append({
        "name": "simulator_checkpoint",
        "pass": os.path.isfile(model_path + ".index") or os.path.isfile(model_path + ".meta"),
        "detail": model_path,
    })
    checks.append({
        "name": "sample_file",
        "pass": os.path.isfile(config["sample_file"]),
        "detail": config["sample_file"],
    })
    checks.append({
        "name": "terminal_reward_nonzero",
        "pass": logged["terminal_nonzero_rate"] > 0.1,
        "detail": "nonzero_rate={:.3f}".format(logged["terminal_nonzero_rate"]),
    })
    checks.append({
        "name": "action_mask_shrinks",
        "pass": logged["mask_shrink_steps"] > 0 or logged["mask_final_mean_valid"] < config["action_size"],
        "detail": "final_mean_valid={:.1f} shrink_steps={}".format(
            logged["mask_final_mean_valid"], logged["mask_shrink_steps"]
        ),
    })
    checks.append({
        "name": "logged_beats_random",
        "pass": logged["return_mean"] > random_policy["return_mean"],
        "detail": "logged={:.2f} random={:.2f}".format(
            logged["return_mean"], random_policy["return_mean"]
        ),
    })
    return {"passed": all(c["pass"] for c in checks), "checks": checks}


def write_summary_md(path, payload, gate):
    lines = [
        "# Phase 0: Online SlateRecEnv Smoke",
        "",
        "Generated: `{}`".format(payload["generated_at"]),
        "",
        "## Config",
        "",
        "- model: `{}`".format(payload["config"]["model_file"]),
        "- sample: `{}`".format(payload["config"]["sample_file"]),
        "- batch_size: {}".format(payload["config"]["batch_size"]),
        "",
        "| Policy | Mean | Std | Terminal nonzero rate |",
        "|--------|-----:|----:|----------------------:|",
    ]
    for key in ("logged", "random_masked"):
        r = payload["results"][key]
        lines.append("| {} | {:.2f} | {:.2f} | {:.3f} |".format(
            key, r["return_mean"], r["return_std"], r["terminal_nonzero_rate"]))
    lines.extend([
        "",
        "## Gate thresholds (Phase 2)",
        "",
        "- abs_sim_gate: **{:.2f}**".format(payload["gate_thresholds"]["abs_sim_gate"]),
        "- rel_sim_gate: **{:.1f}%**".format(100 * payload["gate_thresholds"]["rel_sim_gate"]),
        "",
        "## Phase 0 gate: **{}**".format("PASS" if gate["passed"] else "FAIL"),
        "",
    ])
    for c in gate["checks"]:
        lines.append("- [{}] {} - {}".format("ok" if c["pass"] else "FAIL", c["name"], c["detail"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 online env smoke test")
    parser.add_argument("--num-episodes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Parallel env slots; use 1-8 if RAM is limited")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    configure_runtime(force_cpu=args.cpu)

    output_dir = os.environ.get("rl4rs_output_dir", os.path.join(ROOT, "output"))
    dataset_dir = os.environ.get("rl4rs_dataset_dir", os.path.join(ROOT, "dataset"))
    pilot_dir = os.path.join(output_dir, "qlearning_pilot")
    os.makedirs(pilot_dir, exist_ok=True)

    config = build_online_slate_config(
        output_dir, dataset_dir, batch_size=args.batch_size,
        gpu=simulator_config_gpu(use_cpu=args.cpu),
    )

    print("Phase 0 smoke", flush=True)
    print("  model:", config["model_file"], flush=True)
    print("  sample:", config["sample_file"], flush=True)
    print("  batch_size:", args.batch_size, "episodes:", args.num_episodes, flush=True)
    print("  tf_sim_gpu:", not config.get("gpu", True), flush=True)

    env = make_slate_env(config)
    logged = run_policy(env, "logged", args.num_episodes, args.seed, config)
    print("Running random masked policy...", flush=True)
    random_res = run_policy(env, "random_masked", args.num_episodes, args.seed + 1, config)

    gate_thresholds = compute_gate_thresholds(logged["return_mean"])
    gate = evaluate_phase0_gate(logged, random_res, config)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "phase": 0,
        "config": {k: config[k] for k in (
            "env", "model_file", "sample_file", "batch_size", "max_steps",
            "support_rllib_mask", "support_conti_env",
        )},
        "results": {"logged": logged, "random_masked": random_res},
        "gate_thresholds": gate_thresholds,
        "phase0_gate": gate,
    }

    json_path = os.path.join(pilot_dir, "baseline_rewards.json")
    md_path = os.path.join(pilot_dir, "phase0_smoke_summary.md")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    write_summary_md(md_path, payload, gate)

    print("Logged mean={:.2f} | Random mean={:.2f} | Gate {}".format(
        logged["return_mean"], random_res["return_mean"], "PASS" if gate["passed"] else "FAIL"), flush=True)
    print("Wrote", json_path, flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
