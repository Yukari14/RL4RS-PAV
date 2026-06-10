#!/usr/bin/env python3
"""
Official RLlib PPO pilot (modelfree_train.py settings) with optional online PAV.

Mirrors script/modelfree_train.py PPO + reproductions/run_modelfree_rl.sh SlateRecEnv-v0/a_all.
Uses local SlateRecEnv (same sim as gymHttpServer) instead of HttpEnv for PAV compatibility.
"""
import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime

import gym
import numpy as np
import ray
import tensorflow as tf
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.nets.rllib.rllib_mask_model import getMaskActionsModel
from rl4rs.online.config import default_pav_config, load_gate_thresholds
from rl4rs.online.pav_cli import add_pav_cli_args, apply_pav_cli_overrides, pav_condition_tag
from rl4rs.online.env_utils import attach_slate_masks
from rl4rs.pav.online import PAVRewardWrapper, load_pav_artifacts
from rl4rs.policy.policy_model import policy_model
from rl4rs.utils.rllib_print import pretty_print
from rl4rs.utils.rllib_vector_env import MyVectorEnvWrapper
from script.modelfree_trainer import get_rl_model


def build_official_slate_config(output_dir, dataset_dir, batch_size=64, trial_name="a_all", epoch=100):
    """Same fields as modelfree_train.py defaults + run_modelfree_rl.sh overrides."""
    return {
        "epoch": epoch,
        "maxlen": 64,
        "batch_size": batch_size,
        "action_size": 284,
        "class_num": 2,
        "dense_feature_num": 432,
        "category_feature_num": 21,
        "category_hash_size": 100000,
        "seq_num": 2,
        "emb_size": 128,
        "is_eval": False,
        "hidden_units": 128,
        "max_steps": 9,
        "action_emb_size": 32,
        "page_items": 9,
        "sample_file": os.path.join(dataset_dir, "rl4rs_dataset_a_shuf.csv"),
        "model_file": os.path.join(output_dir, "simulator_a_dien", "model"),
        "iteminfo_file": os.path.join(dataset_dir, "item_info.csv"),
        "support_rllib_mask": True,
        "env": "SlateRecEnv-v0",
        "trial_name": trial_name,
        "gpu": True,
    }


def build_official_ppo_rllib_config(config):
    """PPO block from modelfree_train.py merged with common rllib_config."""
    ppo_cfg = {
        "num_workers": 2,
        "use_critic": True,
        "use_gae": True,
        "lambda": 1.0,
        "kl_coeff": 0.2,
        "sgd_minibatch_size": 256,
        "shuffle_sequences": True,
        "num_sgd_iter": 1,
        "lr": 0.0001,
        "vf_loss_coeff": 0.5,
        "clip_param": 0.3,
        "vf_clip_param": 500.0,
        "kl_target": 0.01,
        "model": {
            "vf_share_layers": False,
            "custom_model": "mask_model",
        },
    }
    return dict(
        {
            "env": "rllibEnv-v0",
            "gamma": 1,
            "explore": True,
            "exploration_config": {"type": "SoftQ"},
            "num_gpus": 1 if config.get("gpu", True) else 0,
            "num_workers": 0,
            "framework": "tf",
            "rollout_fragment_length": config["max_steps"],
            "batch_mode": "complete_episodes",
            "train_batch_size": min(config["batch_size"] * config["max_steps"], 1024),
            "evaluation_interval": 500,
            "evaluation_num_episodes": 2048 * 4,
            "evaluation_config": {"explore": False},
            "log_level": "INFO",
        },
        **ppo_cfg,
    )


def make_official_vector_env(env_config, artifacts=None):
    """Local SlateRecEnv; sim gpu=False like gymHttpServer."""
    cfg = deepcopy(env_config)
    attach_slate_masks(cfg)
    cfg["gpu"] = False
    sim = SlateRecEnv(cfg, state_cls=SlateState)
    env = gym.make("SlateRecEnv-v0", recsim=sim)
    if artifacts is not None:
        env = PAVRewardWrapper(env, artifacts=artifacts, enabled=True)
    return MyVectorEnvWrapper(env, cfg["batch_size"])


def make_pav_env_creator(env_config, pav_config=None):
    """Load PAV inside each Ray worker on CPU (CUDA tensors cannot be pickled to workers)."""
    cache = {}

    def _creator(_):
        artifacts = None
        if pav_config is not None:
            if "artifacts" not in cache:
                worker_cfg = deepcopy(pav_config)
                worker_cfg.device = "cpu"
                cache["artifacts"] = load_pav_artifacts(worker_cfg)
            artifacts = cache["artifacts"]
        return make_official_vector_env(env_config, artifacts=artifacts)

    return _creator


def eval_sim_reward_v2(trainer, base_config, num_batches=4):
    """Official modelfree_train.py eval_v2 on local raw SlateRecEnv."""
    cfg = deepcopy(base_config)
    cfg["is_eval"] = True
    cfg["batch_size"] = 2048
    cfg["cache_size"] = 2048
    cfg["gpu"] = False
    attach_slate_masks(cfg)
    sim = SlateRecEnv(cfg, state_cls=SlateState)
    eval_env = gym.make("SlateRecEnv-v0", recsim=sim)
    policy = policy_model(trainer, cfg)
    episode_reward = 0.0
    all_actions = []
    for _ in range(num_batches):
        obs = eval_env.reset()
        for _ in range(cfg["max_steps"]):
            action = np.array(policy.action_probs(obs)).argmax(axis=1)
            obs, reward, _done, _info = eval_env.step(action)
            episode_reward += float(np.sum(reward))
            all_actions.append(action)
    avg = episode_reward / cfg["batch_size"] / num_batches
    flat = np.concatenate(all_actions, axis=0)
    return {
        "sim_avg_reward": avg,
        "n_eval_batches": num_batches,
        "eval_batch_size": cfg["batch_size"],
        "action_diversity": {
            "unique_actions": int(len(np.unique(flat))),
            "total_actions": int(flat.size),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Official RLlib PPO (modelfree_train) + optional online PAV"
    )
    parser.add_argument("--use-pav", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Official modelfree default is 10000")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--trial-name", default="a_all",
                        help="Official trial_name; PAV run uses <trial>_pav_v2")
    add_pav_cli_args(parser)
    args = parser.parse_args()

    output_dir = os.environ.get("rl4rs_output_dir", os.path.join(ROOT, "output"))
    dataset_dir = os.environ.get("rl4rs_dataset_dir", os.path.join(ROOT, "dataset"))
    pilot_dir = os.path.join(output_dir, "ppo_pilot")
    os.makedirs(pilot_dir, exist_ok=True)

    trial_name = args.trial_name
    if args.use_pav:
        trial_name = "{}_{}".format(args.trial_name, args.pav_suffix)

    config = build_official_slate_config(
        output_dir, dataset_dir,
        batch_size=args.batch_size,
        trial_name=trial_name,
        epoch=args.epochs,
    )

    cond = pav_condition_tag(args, use_pav=args.use_pav)
    modelfile = "PPO_{}_seed{}_{}".format(config["env"], args.seed, trial_name)
    checkpoint_dir = os.path.join(output_dir, "ray_results", modelfile)
    os.makedirs(checkpoint_dir, exist_ok=True)

    env_config = deepcopy(config)
    pav_config = None
    if args.use_pav:
        pav_config = default_pav_config(
            output_dir, dataset_dir, trial_name=args.pav_trial_name, suffix=args.pav_suffix
        )
        pav_config = apply_pav_cli_overrides(pav_config, args)

    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
        object_store_memory=int(2e9),
    )

    mask_model = getMaskActionsModel(true_obs_shape=(256,), action_size=config["action_size"])
    ModelCatalog.register_custom_model("mask_model", mask_model)

    register_env(
        "rllibEnv-v0",
        make_pav_env_creator(env_config, pav_config=pav_config),
    )

    rllib_config = build_official_ppo_rllib_config(config)
    print(
        "Training official PPO cond={} trial={} seed={} epochs={} "
        "batch={} num_workers={} num_gpus={}".format(
            cond, trial_name, args.seed, args.epochs, args.batch_size,
            rllib_config["num_workers"], rllib_config["num_gpus"],
        ),
        flush=True,
    )

    trainer = get_rl_model("PPO", rllib_config)
    last_result = None
    for i in range(args.epochs):
        last_result = trainer.train()
        if (i + 1) % args.log_every == 0 or i == 0:
            print("epoch {} {}".format(i + 1, pretty_print(last_result)), flush=True)
        if (i + 1) % 500 == 0:
            ckpt = trainer.save(checkpoint_dir=checkpoint_dir)
            print("checkpoint saved at", ckpt, flush=True)

    ckpt = trainer.save(checkpoint_dir=checkpoint_dir)
    print("final checkpoint", ckpt, flush=True)

    eval_metrics = eval_sim_reward_v2(trainer, config)
    gate = load_gate_thresholds(os.path.join(output_dir, "qlearning_pilot"))
    logged_mean = gate.get("logged_mean")
    delta_vs_logged = None
    if logged_mean is not None:
        delta_vs_logged = eval_metrics["sim_avg_reward"] - float(logged_mean)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "algorithm": "PPO",
        "implementation": "modelfree_train_official",
        "condition": cond,
        "trial_name": trial_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_dir": checkpoint_dir,
        "final_checkpoint": ckpt,
        "rllib": {
            "num_workers": rllib_config["num_workers"],
            "num_gpus": rllib_config["num_gpus"],
            "train_batch_size": rllib_config["train_batch_size"],
        },
        "pav_config": {
            k: getattr(pav_config, k)
            for k in (
                "alpha", "k", "directional_lambda", "use_verifier", "use_raw_progress",
            )
        } if pav_config is not None else None,
        "train_tail": {
            k: last_result.get(k)
            for k in ("episode_reward_mean", "episode_len_mean", "timesteps_total")
            if last_result and k in last_result
        },
        "eval_raw_sim": eval_metrics,
        "gate_thresholds": gate,
        "delta_vs_logged_mean": delta_vs_logged,
    }
    summary_path = os.path.join(pilot_dir, "ppo_official_{}_seed{}_summary.json".format(cond, args.seed))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Eval sim_avg_reward={:.2f} unique_actions={}".format(
        eval_metrics["sim_avg_reward"],
        eval_metrics["action_diversity"]["unique_actions"],
    ), flush=True)
    if logged_mean is not None:
        print("logged_mean={:.2f} delta={:+.2f}".format(logged_mean, delta_vs_logged or 0.0), flush=True)
    print("Saved", summary_path, flush=True)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
