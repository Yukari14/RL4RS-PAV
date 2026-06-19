#!/usr/bin/env python3
"""Train full frozen PAV artifact suites (prover + reward + verifier) per prover_kind.

Each variant gets its own suffix under output/pav/:
  stats_*, Reward_*, Verifier_*, Policy_* (supervised/bo_k only), optional shaped .h5

RL comparison later:
  python script/dqn_pav_pilot.py --use-pav --pav-suffix pav_v3_sup ...
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from rl4rs.online.config import default_pav_config
from rl4rs.pav.dataset import export_mdpdataset, is_discrete_actions, load_mdpdataset
from rl4rs.pav.diagnostics import format_prover_quality
from rl4rs.pav.prover import format_prover_banner
from rl4rs.pav.suite_progress import clear as clear_suite_progress
from rl4rs.pav.suite_progress import configure as configure_suite_progress
from rl4rs.pav.suite_progress import default_paths as suite_progress_paths
from rl4rs.pav.suite_progress import heartbeat as suite_heartbeat
from rl4rs.pav.suite_progress import note as suite_note
from rl4rs.pav.suite_progress import print_watch_banner
from rl4rs.pav.trainer import build_pav_signals

PROVER_VARIANTS = [
    {"prover_kind": "logging", "suffix": "pav_v3_log", "label": "logging (offline actions)"},
    {"prover_kind": "supervised", "suffix": "pav_v3_sup", "label": "supervised BC prover"},
    {"prover_kind": "bo_k", "suffix": "pav_v3_bok", "label": "bo_k BC prover", "prover_bo_k": 3},
    {
        "prover_kind": "random",
        "suffix": "pav_v3_rand",
        "label": "random masked prover",
        "config_overrides": {"max_mc_states": 300, "n_mc": 4},
    },
    {
        "prover_kind": "uniform",
        "suffix": "pav_v3_unif",
        "label": "uniform masked prover",
        "config_overrides": {"max_mc_states": 300, "n_mc": 4},
    },
]

FAST_MC_DEFAULTS = {
    "max_mc_states": 500,
    "n_mc": 4,
    "mc_max_workers": 4,
    "mc_sim_batch_size": 1,
    "mc_sim_use_cpu": True,
    "mc_progress_every": 25,
    "hybrid_mc_fraction": 0.1,
    "reward_epochs": 10,
    "verifier_epochs": 8,
}


def _artifact_paths(config):
    prefix = config.artifact_prefix
    out = config.pav_output_dir
    paths = {
        "stats": config.stats_path,
        "reward": config.reward_model_path,
        "verifier": config.verifier_path,
        "policy": os.path.join(out, "Policy_{}.pt".format(prefix)),
        "shaped_h5": config.shaped_dataset_path,
    }
    return paths


def _suite_complete(paths, require_policy=False):
    required = [paths["stats"], paths["reward"]]
    if require_policy:
        required.append(paths["policy"])
    if os.path.isfile(paths["stats"]):
        try:
            with open(paths["stats"]) as f:
                stats = json.load(f)
            if stats.get("verifier_metrics") and paths["verifier"] and not os.path.isfile(paths["verifier"]):
                return False
        except (IOError, json.JSONDecodeError):
            return False
    return all(os.path.isfile(p) for p in required if p)


def _build_config(variant, output_dir, dataset_dir, trial_name, mc_overrides=None):
    suffix = variant["suffix"]
    kind = variant["prover_kind"]
    overrides = dict(FAST_MC_DEFAULTS)
    overrides.update(variant.get("config_overrides") or {})
    if mc_overrides:
        overrides.update(mc_overrides)
    overrides["prover_kind"] = kind
    if "prover_bo_k" in variant:
        overrides["prover_bo_k"] = int(variant["prover_bo_k"])
    return default_pav_config(
        output_dir, dataset_dir, trial_name=trial_name, suffix=suffix, **overrides
    )


def train_one_variant(variant, output_dir, dataset_dir, trial_name, export_shaped, force, index, total, mc_overrides):
    config = _build_config(variant, output_dir, dataset_dir, trial_name, mc_overrides)
    suffix = variant["suffix"]
    kind = variant["prover_kind"]
    paths = _artifact_paths(config)
    need_policy = kind in ("supervised", "bo_k")

    configure_suite_progress(output_dir, index, total, suffix, kind)

    if not force and _suite_complete(paths, require_policy=need_policy):
        suite_note("skip", status="skipped", reason="artifacts_exist")
        print("[skip] {} prover_kind={} artifacts already exist".format(suffix, kind), flush=True)
        return {"suffix": suffix, "prover_kind": kind, "skipped": True, "paths": paths}

    if not os.path.isfile(config.raw_dataset_path):
        raise FileNotFoundError("Raw dataset missing: {}".format(config.raw_dataset_path))

    print("", flush=True)
    print("=" * 72, flush=True)
    print("PAV suite: {} | prover_kind={}".format(suffix, kind), flush=True)
    print(
        "  MC: states={} n_mc={} workers={} sim={} reward_ep={} verifier_ep={}".format(
            config.max_mc_states,
            config.n_mc,
            config.mc_max_workers,
            "cpu" if getattr(config, "mc_sim_use_cpu", False) else "gpu",
            config.reward_epochs,
            config.verifier_epochs,
        ),
        flush=True,
    )
    for line in format_prover_banner(config):
        print("  " + line, flush=True)
    print("=" * 72, flush=True)

    t0 = time.time()
    dataset = load_mdpdataset(config.raw_dataset_path)
    suite_note("load_dataset")
    stop_event = threading.Event()

    def _heartbeat_loop():
        while not stop_event.wait(30.0):
            suite_heartbeat()

    hb = threading.Thread(target=_heartbeat_loop, name="pav-suite-heartbeat", daemon=True)
    hb.start()
    try:
        signals = build_pav_signals(dataset, config)
    finally:
        stop_event.set()
        hb.join(timeout=1.0)
    stats = signals["stats"]

    if export_shaped:
        flat = signals["flat"]
        export_mdpdataset(
            flat["observations"],
            flat["actions"],
            signals["shaped_rewards"],
            flat["terminals"],
            config.shaped_dataset_path,
            discrete_action=is_discrete_actions(flat["actions"]),
        )
        print("Shaped dataset -> {}".format(config.shaped_dataset_path), flush=True)

    elapsed = time.time() - t0
    for line in format_prover_quality(stats):
        print(line, flush=True)
    print(
        "Saved stats={} reward={} verifier={}".format(
            paths["stats"], paths["reward"], paths["verifier"]
        ),
        flush=True,
    )
    if need_policy and os.path.isfile(paths["policy"]):
        print("Policy -> {}".format(paths["policy"]), flush=True)

    suite_note("variant_complete", status="complete", elapsed_sec=round(elapsed, 1))

    row = {
        "suffix": suffix,
        "prover_kind": kind,
        "label": variant.get("label", kind),
        "skipped": False,
        "elapsed_sec": round(elapsed, 1),
        "mc_config": {
            "max_mc_states": config.max_mc_states,
            "n_mc": config.n_mc,
            "mc_max_workers": config.mc_max_workers,
            "reward_epochs": config.reward_epochs,
            "verifier_epochs": config.verifier_epochs,
        },
        "paths": paths,
        "stats_summary": {
            "contribution_return_corr": stats.get("contribution_return_corr"),
            "distinguishability": stats.get("distinguishability"),
            "alignment_corr": stats.get("alignment_corr"),
            "value_mse": (stats.get("reward_metrics") or {}).get("value_mse"),
            "verifier_q_mse": (stats.get("verifier_metrics") or {}).get("verifier_q_mse"),
            "verifier_auc": (stats.get("verifier_metrics") or {}).get("verifier_auc"),
            "actions_source": stats.get("actions_source"),
            "prover_artifact_path": stats.get("prover_artifact_path"),
        },
        "dqn_cli": (
            "python script/dqn_pav_pilot.py --use-pav --pav-suffix {} "
            "--pav-trial-name {} --epochs 150 --seed 0"
        ).format(suffix, trial_name),
    }
    return row


def _parse_mc_overrides(args):
    overrides = {}
    if args.max_mc_states is not None:
        overrides["max_mc_states"] = int(args.max_mc_states)
    if args.n_mc is not None:
        overrides["n_mc"] = int(args.n_mc)
    if args.mc_workers is not None:
        overrides["mc_max_workers"] = int(args.mc_workers)
    if args.reward_epochs is not None:
        overrides["reward_epochs"] = int(args.reward_epochs)
    if args.verifier_epochs is not None:
        overrides["verifier_epochs"] = int(args.verifier_epochs)
    if args.mc_sim_gpu:
        overrides["mc_sim_use_cpu"] = False
    elif args.mc_sim_cpu:
        overrides["mc_sim_use_cpu"] = True
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Train full PAV artifact suite per prover_kind")
    parser.add_argument("--trial-name", default="a_50k_logged")
    parser.add_argument("--output-dir", default=os.environ.get("rl4rs_output_dir", os.path.join(ROOT, "output")))
    parser.add_argument("--dataset-dir", default=os.environ.get("rl4rs_dataset_dir", os.path.join(ROOT, "dataset")))
    parser.add_argument(
        "--kinds",
        default="logging,supervised,bo_k",
        help="Comma list or 'all' (default: core 3 provers)",
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if artifacts exist")
    parser.add_argument("--no-shaped-h5", action="store_true", help="Skip shaped .h5 export")
    parser.add_argument("--max-mc-states", type=int, default=None)
    parser.add_argument("--n-mc", type=int, default=None)
    parser.add_argument("--mc-workers", type=int, default=None)
    parser.add_argument("--mc-sim-cpu", action="store_true", default=None,
                        help="Force CPU DIEN sim during MC (default: True for suite script)")
    parser.add_argument("--mc-sim-gpu", action="store_true", help="Use GPU DIEN sim during MC")
    parser.add_argument("--reward-epochs", type=int, default=None)
    parser.add_argument("--verifier-epochs", type=int, default=None)
    args = parser.parse_args()

    if args.kinds.strip().lower() == "all":
        variants = list(PROVER_VARIANTS)
    else:
        wanted = {k.strip() for k in args.kinds.split(",")}
        variants = [v for v in PROVER_VARIANTS if v["prover_kind"] in wanted]
        if not variants:
            raise SystemExit("No matching prover kinds in: {}".format(args.kinds))

    mc_overrides = _parse_mc_overrides(args)
    manifest_path = os.path.join(args.output_dir, "pav", "prover_suites_manifest.json")
    paths = suite_progress_paths(args.output_dir)
    from rl4rs.pav.suite_progress import reset_suite_timer
    reset_suite_timer()
    clear_suite_progress(paths["live"])
    clear_suite_progress(paths["current"])
    print_watch_banner(args.output_dir)
    rows = []
    total = len(variants)
    for idx, variant in enumerate(variants, start=1):
        try:
            rows.append(
                train_one_variant(
                    variant,
                    args.output_dir,
                    args.dataset_dir,
                    args.trial_name,
                    export_shaped=not args.no_shaped_h5,
                    force=args.force,
                    index=idx,
                    total=total,
                    mc_overrides=mc_overrides,
                )
            )
        except Exception as exc:
            suite_note("fail", status="failed", error=str(exc))
            print("[FAIL] {} prover_kind={}: {}".format(variant["suffix"], variant["prover_kind"], exc), flush=True)
            rows.append(
                {
                    "suffix": variant["suffix"],
                    "prover_kind": variant["prover_kind"],
                    "error": str(exc),
                }
            )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "trial_name": args.trial_name,
        "mc_defaults": dict(FAST_MC_DEFAULTS),
        "cli_mc_overrides": mc_overrides,
        "variants": rows,
        "note": "Each suffix is a frozen triple (prover metadata + reward + verifier). "
                "Pick --pav-suffix for DQN comparison.",
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("", flush=True)
    print("Manifest -> {}".format(manifest_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
