#!/usr/bin/env python3
"""
Official RLlib DQN pilot (modelfree_train.py settings) with optional online PAV.

Mirrors script/modelfree_train.py DQN + reproductions/run_modelfree_rl.sh SlateRecEnv-v0/a_all.
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
from rl4rs.pav.online import (
    PAVRewardWrapper,
    load_pav_artifacts,
    print_pav_artifact_summary,
    print_pav_online_startup,
)
from rl4rs.pav.training_progress import LiveProgressBoard
from rl4rs.policy.policy_model import policy_model
from rl4rs.utils.rllib_print import pretty_print
from rl4rs.utils.rllib_vector_env import MyVectorEnvWrapper
from script.modelfree_trainer import get_rl_model


def build_official_slate_config(output_dir, dataset_dir, batch_size=64, trial_name="a_all", epoch=10000):
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


def _configure_ray_quiet():
    """Reduce Ray dashboard/prometheus noise on headless training boxes."""
    import logging

    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    for name in ("ray", "ray.worker", "ray.tune", "ray.rllib", "ray.util"):
        logging.getLogger(name).setLevel(logging.ERROR)

    class _RayNoiseFilter(logging.Filter):
        _needles = (
            "socket.gaierror",
            "prometheus_exporter",
            "metrics_agent",
            "The agent on node",
            "new_dashboard",
            "ray.new_dashboard",
        )

        def filter(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            return not any(n in msg for n in self._needles)

    flt = _RayNoiseFilter()
    logging.getLogger().addFilter(flt)
    logging.getLogger("ray").addFilter(flt)


def build_official_dqn_rllib_config(config, rllib_eval_every=0):
    dqn_cfg = {
        "hiddens": [],
        "dueling": False,
        "double_q": True,
        "n_step": 1,
        "target_network_update_freq": 200,
        "buffer_size": 100000,
        "model": {"custom_model": "mask_model"},
    }
    eval_interval = None if int(rllib_eval_every) <= 0 else max(1, int(rllib_eval_every))
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
            "evaluation_interval": eval_interval,
            "evaluation_num_episodes": 2048 * 4,
            "evaluation_config": {"explore": False},
            "log_level": "INFO",
        },
        **dqn_cfg,
    )


def make_official_vector_env(env_config, artifacts=None, monitor_paths=None):
    cfg = deepcopy(env_config)
    attach_slate_masks(cfg)
    cfg["gpu"] = False
    sim = SlateRecEnv(cfg, state_cls=SlateState)
    env = gym.make("SlateRecEnv-v0", recsim=sim)
    if artifacts is not None:
        artifacts["slate_env_config"] = deepcopy(cfg)
        monitor_paths = monitor_paths or {}
        env = PAVRewardWrapper(
            env,
            artifacts=artifacts,
            enabled=True,
            monitor_log_path=monitor_paths.get("pav_monitor_csv"),
            monitor_state_path=monitor_paths.get("pav_state_json"),
        )
    return MyVectorEnvWrapper(env, cfg["batch_size"])


def make_pav_env_creator(env_config, pav_config=None, monitor_paths=None):
    """DQN uses num_workers=0; load once in driver (CUDA ok)."""
    cache = {"artifacts": None}

    def _creator(_):
        artifacts = None
        if pav_config is not None:
            if cache["artifacts"] is None:
                worker_cfg = deepcopy(pav_config)
                import torch
                worker_cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                cache["artifacts"] = load_pav_artifacts(worker_cfg)
            artifacts = cache["artifacts"]
        return make_official_vector_env(
            env_config, artifacts=artifacts, monitor_paths=monitor_paths
        )

    return _creator, cache


def _enrich_pav_state(pav_state, artifacts):
    """Attach frozen offline prover_kind for progress logging."""
    out = dict(pav_state or {})
    if artifacts:
        stats = artifacts.get("stats") or {}
        cfg = artifacts.get("config")
        out["prover_kind"] = stats.get(
            "prover_kind",
            getattr(cfg, "prover_kind", "logging") if cfg is not None else "logging",
        )
    return out


def _curve_row(epoch, result, sim_eval=None, pav_state=None):
    row = {
        "epoch": int(epoch),
        "train_episode_reward_mean": float(result.get("episode_reward_mean", float("nan"))),
        "train_episode_len_mean": float(result.get("episode_len_mean", float("nan"))),
        "timesteps_total": int(result.get("timesteps_total", 0)),
    }
    evaluation = result.get("evaluation") or {}
    if evaluation.get("episode_reward_mean") is not None:
        row["rllib_eval_episode_reward_mean"] = float(evaluation["episode_reward_mean"])
    if sim_eval is not None:
        row["sim_eval_avg_reward"] = float(sim_eval.get("sim_avg_reward", float("nan")))
    if pav_state:
        if pav_state.get("prover_kind"):
            row["prover_kind"] = str(pav_state["prover_kind"])
        if pav_state.get("alpha_scale") is not None:
            row["pav_alpha_scale"] = float(pav_state["alpha_scale"])
        if pav_state.get("rolling_distinguishability") is not None:
            row["pav_rolling_distinguishability"] = float(pav_state["rolling_distinguishability"])
    return row


def _print_curve_row(row):
    parts = [
        "epoch={}".format(row["epoch"]),
        "train_reward_mean={:.4f}".format(row["train_episode_reward_mean"]),
    ]
    if "rllib_eval_episode_reward_mean" in row:
        parts.append("rllib_eval={:.4f}".format(row["rllib_eval_episode_reward_mean"]))
    if "sim_eval_avg_reward" in row:
        parts.append("sim_eval={:.4f}".format(row["sim_eval_avg_reward"]))
    if "pav_alpha_scale" in row:
        parts.append("alpha_scale={:.3f}".format(row["pav_alpha_scale"]))
    if "pav_rolling_distinguishability" in row:
        parts.append("pav_dist={:.4f}".format(row["pav_rolling_distinguishability"]))
    if "prover_kind" in row:
        parts.append("prover_kind={}".format(row["prover_kind"]))
    parts.append("timesteps={}".format(row["timesteps_total"]))
    print("[reward_curve] " + " ".join(parts), flush=True)


def _save_curve_csv(path, rows):
    if not rows:
        return
    import csv
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def eval_sim_reward_v2(trainer, base_config, num_batches=4, sim_on_cpu=False, eval_batch_size=2048):
    cfg = deepcopy(base_config)
    cfg["is_eval"] = True
    cfg["batch_size"] = eval_batch_size
    cfg["cache_size"] = eval_batch_size
    # RecSimBase inverts gpu: True -> CPU session, False -> GPU session.
    cfg["gpu"] = True if sim_on_cpu else False
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


def _read_curve_tail(curve_path):
    if not os.path.isfile(curve_path):
        return None
    import csv
    with open(curve_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def run_eval_only(args, config, cond, trial_name, checkpoint_dir, pilot_dir, pav_config, monitor_paths, board):
    ckpt = args.eval_only_checkpoint
    if ckpt is None:
        raise ValueError("--eval-only-checkpoint is required")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("checkpoint not found: {}".format(ckpt))

    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    _configure_ray_quiet()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
        logging_level="ERROR",
        object_store_memory=int(2e9),
    )

    mask_model = getMaskActionsModel(true_obs_shape=(256,), action_size=config["action_size"])
    ModelCatalog.register_custom_model("mask_model", mask_model)
    env_config = deepcopy(config)
    register_env(
        "rllibEnv-v0",
        make_pav_env_creator(env_config, pav_config=pav_config, monitor_paths=monitor_paths)[0],
    )

    rllib_config = build_official_dqn_rllib_config(config, rllib_eval_every=0)
    rllib_config["num_gpus"] = 0
    trainer = get_rl_model("DQN", rllib_config)
    print("Eval-only: restoring {}".format(ckpt), flush=True)
    trainer.restore(ckpt)

    eval_metrics = eval_sim_reward_v2(
        trainer, config, sim_on_cpu=True, eval_batch_size=512, num_batches=16,
    )
    gate = load_gate_thresholds(os.path.join(os.environ.get("rl4rs_output_dir", os.path.join(ROOT, "output")), "qlearning_pilot"))
    curve_path = board.paths["curve_csv"]
    curve_tail = _read_curve_tail(curve_path)
    logged_mean = gate.get("logged_mean")
    delta_vs_logged = None
    if logged_mean is not None:
        delta_vs_logged = eval_metrics["sim_avg_reward"] - float(logged_mean)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "algorithm": "DQN",
        "implementation": "modelfree_train_official",
        "condition": cond,
        "trial_name": trial_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_dir": checkpoint_dir,
        "final_checkpoint": ckpt,
        "note": "eval-only from checkpoint (sim eval; RLlib eval skipped)",
        "rllib": {
            "num_workers": rllib_config["num_workers"],
            "num_gpus": rllib_config["num_gpus"],
            "train_batch_size": rllib_config["train_batch_size"],
        },
        "pav_config": {
            k: getattr(pav_config, k)
            for k in ("alpha", "k", "directional_lambda", "use_verifier", "use_raw_progress")
        } if pav_config is not None else None,
        "train_tail": {
            k: curve_tail.get(k) if curve_tail else None
            for k in ("train_episode_reward_mean", "train_episode_len_mean", "timesteps_total")
        } if curve_tail else None,
        "reward_curve_csv": curve_path,
        "reward_curve_tail": curve_tail,
        "eval_rllib": {},
        "eval_raw_sim": eval_metrics,
        "gate_thresholds": gate,
        "delta_vs_logged_mean": delta_vs_logged,
    }
    summary_path = os.path.join(pilot_dir, "dqn_official_{}_seed{}_summary.json".format(cond, args.seed))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Eval sim_avg_reward={:.2f} unique_actions={}".format(
        eval_metrics["sim_avg_reward"],
        eval_metrics["action_diversity"]["unique_actions"],
    ), flush=True)
    print("Saved", summary_path, flush=True)
    ray.shutdown()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Official RLlib DQN (modelfree_train) + optional online PAV"
    )
    parser.add_argument("--use-pav", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Official modelfree default is 10000")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print training metrics every N epochs (default: every epoch)")
    parser.add_argument("--sim-eval-every", type=int, default=0,
                        help="Raw simulator eval every N epochs (0=only at end; slow)")
    parser.add_argument("--rllib-eval-every", type=int, default=0,
                        help="RLlib built-in evaluation every N epochs (0=only at end)")
    parser.add_argument("--trial-name", default="a_all")
    parser.add_argument(
        "--eval-only-checkpoint",
        default=None,
        help="Skip training; restore this checkpoint and run sim eval only (CPU trainer, no RLlib eval)",
    )
    add_pav_cli_args(parser)
    args = parser.parse_args()

    output_dir = os.environ.get("rl4rs_output_dir", os.path.join(ROOT, "output"))
    dataset_dir = os.environ.get("rl4rs_dataset_dir", os.path.join(ROOT, "dataset"))
    pilot_dir = os.path.join(output_dir, "dqn_pilot")
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
    modelfile = "DQN_{}_seed{}_{}".format(config["env"], args.seed, trial_name)
    checkpoint_dir = os.path.join(output_dir, "ray_results", modelfile)
    os.makedirs(checkpoint_dir, exist_ok=True)

    env_config = deepcopy(config)
    pav_config = None
    artifacts = None
    monitor_paths = None
    board = LiveProgressBoard(
        pilot_dir, cond, args.seed, args.epochs, use_pav=args.use_pav
    )
    # Reset live log for this run
    open(board.paths["live"], "w").close()

    if args.use_pav:
        pav_config = default_pav_config(
            output_dir, dataset_dir, trial_name=args.pav_trial_name, suffix=args.pav_suffix
        )
        pav_config = apply_pav_cli_overrides(pav_config, args)
        artifacts = load_pav_artifacts(pav_config)
        print_pav_artifact_summary(artifacts, pav_config)
        monitor_paths = {
            "pav_monitor_csv": board.paths.get("pav_monitor_csv"),
            "pav_state_json": board.paths.get("pav_state_json"),
        }
        print_pav_online_startup(artifacts, monitor_state_path=monitor_paths.get("pav_state_json"))

    board.print_watch_banner()

    if args.eval_only_checkpoint:
        return run_eval_only(
            args, config, cond, trial_name, checkpoint_dir, pilot_dir,
            pav_config, monitor_paths, board,
        )

    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    _configure_ray_quiet()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
        logging_level="ERROR",
        object_store_memory=int(2e9),
    )

    mask_model = getMaskActionsModel(true_obs_shape=(256,), action_size=config["action_size"])
    ModelCatalog.register_custom_model("mask_model", mask_model)

    pav_cache = {"artifacts": artifacts if args.use_pav else None}
    env_creator, pav_cache = make_pav_env_creator(
        env_config, pav_config=pav_config, monitor_paths=monitor_paths
    )
    if args.use_pav and artifacts is not None:
        pav_cache["artifacts"] = artifacts
    register_env("rllibEnv-v0", env_creator)

    rllib_config = build_official_dqn_rllib_config(config, rllib_eval_every=args.rllib_eval_every)
    sim_eval_note = (
        "end only" if args.sim_eval_every <= 0
        else "every {} epochs".format(args.sim_eval_every)
    )
    rllib_eval_note = (
        "end only" if args.rllib_eval_every <= 0
        else "every {} epochs".format(args.rllib_eval_every)
    )
    print(
        "Training official DQN cond={} trial={} seed={} epochs={} "
        "batch={} num_workers={} num_gpus={} rllib_eval={} sim_eval={}".format(
            cond, trial_name, args.seed, args.epochs, args.batch_size,
            rllib_config["num_workers"], rllib_config["num_gpus"],
            rllib_eval_note, sim_eval_note,
        ),
        flush=True,
    )

    trainer = get_rl_model("DQN", rllib_config)
    curve_rows = []
    last_result = None
    curve_path = board.paths["curve_csv"]

    try:
        from tqdm import tqdm
        epoch_iter = tqdm(range(args.epochs), desc="DQN {}".format(cond), unit="ep")
    except ImportError:
        epoch_iter = range(args.epochs)

    for i in epoch_iter:
        last_result = trainer.train()
        sim_eval = None
        if args.sim_eval_every > 0 and (i + 1) % args.sim_eval_every == 0:
            sim_eval = eval_sim_reward_v2(trainer, config)
        pav_state = board.read_pav_state() if args.use_pav else None
        if args.use_pav and pav_cache.get("artifacts"):
            pav_state = _enrich_pav_state(pav_state, pav_cache["artifacts"])
        row = _curve_row(i + 1, last_result, sim_eval=sim_eval, pav_state=pav_state)
        curve_rows.append(row)
        board.save_curve_csv(curve_rows)
        board.update(i + 1, row, pav_state=pav_state)
        if (i + 1) % args.log_every == 0 or i == 0:
            _print_curve_row(row)
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(board.format_tqdm_postfix(row, pav_state))
        if (i + 1) % 500 == 0:
            ckpt = trainer.save(checkpoint_dir=checkpoint_dir)
            print("checkpoint saved at", ckpt, flush=True)

    print("Reward curve saved to {}".format(curve_path), flush=True)

    ckpt = trainer.save(checkpoint_dir=checkpoint_dir)
    print("final checkpoint", ckpt, flush=True)

    rllib_eval_result = None
    if args.rllib_eval_every <= 0:
        print("Running final RLlib evaluation...", flush=True)
        rllib_eval_result = trainer.evaluate()
        if rllib_eval_result.get("episode_reward_mean") is not None:
            print(
                "Eval rllib_episode_reward_mean={:.4f}".format(
                    rllib_eval_result["episode_reward_mean"]
                ),
                flush=True,
            )

    eval_metrics = eval_sim_reward_v2(trainer, config)
    gate = load_gate_thresholds(os.path.join(output_dir, "qlearning_pilot"))
    logged_mean = gate.get("logged_mean")
    delta_vs_logged = None
    if logged_mean is not None:
        delta_vs_logged = eval_metrics["sim_avg_reward"] - float(logged_mean)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "algorithm": "DQN",
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
                "prover_kind",
            )
        } if pav_config is not None else None,
        "train_tail": {
            k: last_result.get(k)
            for k in ("episode_reward_mean", "episode_len_mean", "timesteps_total")
            if last_result and k in last_result
        },
        "reward_curve_csv": curve_path,
        "progress_live": board.paths["live"],
        "progress_meta_json": board.paths["meta_json"],
        "pav_monitor_csv": board.paths.get("pav_monitor_csv"),
        "reward_curve_tail": curve_rows[-1] if curve_rows else None,
        "eval_rllib": {
            k: rllib_eval_result.get(k)
            for k in ("episode_reward_mean", "episode_len_mean")
            if rllib_eval_result and k in rllib_eval_result
        } if rllib_eval_result else None,
        "eval_raw_sim": eval_metrics,
        "gate_thresholds": gate,
        "delta_vs_logged_mean": delta_vs_logged,
    }
    summary_path = os.path.join(pilot_dir, "dqn_official_{}_seed{}_summary.json".format(cond, args.seed))
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
