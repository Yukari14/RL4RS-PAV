#!/usr/bin/env python3
"""Masked online Q-learning pilot (PyTorch) on local SlateRecEnv."""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from rl4rs.online.config import build_online_slate_config, default_pav_config, load_gate_thresholds
from rl4rs.online.pav_cli import add_pav_cli_args, apply_pav_cli_overrides, pav_condition_tag
from rl4rs.online.device import configure_runtime, resolve_torch_device, simulator_config_gpu
from rl4rs.online.env_utils import make_slate_env
from rl4rs.online.qlearning import (
    QNetwork,
    eval_greedy,
    save_q_checkpoint,
    train_qlearning,
)
from rl4rs.pav.online import PAVRewardWrapper, load_pav_artifacts

OBS_DIM = 266
ACTION_DIM = 284


def main():
    parser = argparse.ArgumentParser(description="Online masked Q-learning pilot")
    parser.add_argument("stage", choices=["train", "eval"])
    parser.add_argument("--use-pav", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Parallel simulator slots (speeds up env steps)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--hidden", default="256,128")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for PyTorch and simulator")
    parser.add_argument("--gpu-sim", action="store_true",
                        help=argparse.SUPPRESS)  # deprecated no-op
    add_pav_cli_args(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trial-name", default="pilot")
    args = parser.parse_args()

    configure_runtime(force_cpu=args.cpu)
    device = resolve_torch_device(force_cpu=args.cpu)
    sim_gpu_flag = simulator_config_gpu(use_cpu=args.cpu)

    output_dir = os.environ.get("rl4rs_output_dir", os.path.join(ROOT, "output"))
    dataset_dir = os.environ.get("rl4rs_dataset_dir", os.path.join(ROOT, "dataset"))
    pilot_dir = os.path.join(output_dir, "qlearning_pilot")
    os.makedirs(pilot_dir, exist_ok=True)

    config = build_online_slate_config(
        output_dir, dataset_dir, batch_size=args.batch_size, gpu=sim_gpu_flag
    )
    hidden_units = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    raw_env = make_slate_env(config)
    train_env = raw_env
    artifacts = None
    if args.use_pav:
        pav_config = default_pav_config(
            output_dir, dataset_dir, trial_name=args.pav_trial_name, suffix=args.pav_suffix
        )
        pav_config = apply_pav_cli_overrides(pav_config, args)
        if not args.cpu and torch.cuda.is_available():
            pav_config.device = "cuda:0"
        artifacts = load_pav_artifacts(pav_config)
        train_env = PAVRewardWrapper(raw_env, artifacts=artifacts, enabled=True)

    cond = pav_condition_tag(args, use_pav=args.use_pav)
    ckpt_path = args.checkpoint or os.path.join(
        pilot_dir,
        "qlearning_{}_{}_seed{}.pt".format(cond, config["env"], args.seed),
    )

    q_net = QNetwork(OBS_DIM, ACTION_DIM, hidden_units).to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    print("torch_device={} sim_config_gpu={} batch_size={}".format(
        device, sim_gpu_flag, args.batch_size), flush=True)

    if args.stage == "train":
        print("Training Q-learning cond={} seed={}".format(cond, args.seed), flush=True)
        history = train_qlearning(
            train_env,
            q_net,
            optimizer,
            num_episodes=args.num_episodes,
            max_steps=config["max_steps"],
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            device=device,
            env_config=raw_env.config,
            seed=args.seed,
            log_every=args.log_every,
        )
        save_q_checkpoint(ckpt_path, q_net, {
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden_units": hidden_units,
            "use_pav": args.use_pav,
            "seed": args.seed,
            "num_episodes": args.num_episodes,
            "batch_size": args.batch_size,
            "device": str(device),
        })
        eval_config = dict(raw_env.config)
        eval_config["batch_size"] = min(args.batch_size, args.eval_episodes)
        eval_config["cache_size"] = eval_config["batch_size"]
        eval_env = make_slate_env(eval_config)
        eval_metrics = eval_greedy(
            eval_env, q_net, device, config["max_steps"], args.eval_episodes,
            eval_env.config, seed=args.seed,
        )
        summary = {
            "generated_at": datetime.now().isoformat(),
            "stage": "train",
            "condition": cond,
            "seed": args.seed,
            "checkpoint": ckpt_path,
            "device": str(device),
            "batch_size": args.batch_size,
            "train_history_tail": history[-5:],
            "eval": eval_metrics,
            "gate_thresholds": load_gate_thresholds(pilot_dir),
        }
        summary_path = os.path.join(pilot_dir, "qlearning_{}_seed{}_summary.json".format(cond, args.seed))
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("Eval sim_avg_reward={:.2f} unique={}".format(
            eval_metrics["sim_avg_reward"],
            eval_metrics["action_diversity_masked"]["unique_actions"],
        ), flush=True)
        print("Saved", ckpt_path, flush=True)
        print("Saved", summary_path, flush=True)
        return 0

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("checkpoint not found: {}".format(ckpt_path))
    payload = torch.load(ckpt_path, map_location=device)
    meta = payload.get("metadata", {})
    q_net = QNetwork(
        int(meta.get("obs_dim", OBS_DIM)),
        int(meta.get("action_dim", ACTION_DIM)),
        tuple(meta.get("hidden_units", hidden_units)),
    ).to(device)
    q_net.load_state_dict(payload["model"])
    eval_metrics = eval_greedy(
        raw_env, q_net, device, config["max_steps"], args.eval_episodes, raw_env.config, seed=args.seed
    )
    print(json.dumps(eval_metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    sys.exit(main())
