import json
import os


def build_online_slate_config(output_dir, dataset_dir, batch_size=8, gpu=False):
    """Canonical SlateRecEnv config for online Q-learning / PAV pilot."""
    return {
        "epoch": 1,
        "maxlen": 64,
        "batch_size": batch_size,
        "action_size": 284,
        "class_num": 2,
        "dense_feature_num": 432,
        "category_feature_num": 21,
        "category_hash_size": 100000,
        "seq_num": 2,
        "emb_size": 128,
        "page_items": 9,
        "hidden_units": 128,
        "max_steps": 9,
        "sample_file": os.path.join(dataset_dir, "rl4rs_dataset_a_shuf.csv"),
        "iteminfo_file": os.path.join(dataset_dir, "item_info.csv"),
        "model_file": os.path.join(output_dir, "simulator_a_dien", "model"),
        "support_d3rl_mask": True,
        "support_rllib_mask": False,
        "support_conti_env": False,
        "is_eval": True,
        "cache_size": batch_size,
        "env": "SlateRecEnv-v0",
        "gpu": gpu,
    }


def build_dqn_slate_config(output_dir, dataset_dir, batch_size=64, gpu=True):
    """Official RLlib DQN SlateRecEnv config (256-d obs + action_mask dict)."""
    cfg = build_online_slate_config(output_dir, dataset_dir, batch_size=batch_size, gpu=gpu)
    cfg["support_rllib_mask"] = True
    cfg["support_d3rl_mask"] = False
    cfg["cache_size"] = batch_size
    return cfg


def default_pav_config(output_dir, dataset_dir, trial_name="a_50k_logged", suffix="pav", **overrides):
    from rl4rs.pav.config import PAVConfig

    payload = {
        "env": "SlateRecEnv-v0",
        "trial_name": trial_name,
        "suffix": suffix,
        "alpha": 0.05,
        "confidence_gating": True,
        "max_shaping_ratio": 1.5,
        "min_confidence": 0.2,
        "shaping_abs_floor": 5.0,
        "output_dir": output_dir,
        "dataset_dir": dataset_dir,
    }
    payload.update(overrides)
    return PAVConfig.from_dict(payload)


def load_gate_thresholds(pilot_dir):
    """Load Phase 0 baseline json; fall back to conservative defaults."""
    path = os.path.join(pilot_dir, "baseline_rewards.json")
    defaults = {"abs_sim_gate": 5.0, "rel_sim_gate": 0.05, "logged_mean": None}
    if not os.path.isfile(path):
        return defaults
    with open(path, "r") as f:
        payload = json.load(f)
    thresholds = payload.get("gate_thresholds", defaults)
    logged = payload.get("results", {}).get("logged", {})
    thresholds["logged_mean"] = logged.get("return_mean")
    thresholds["random_mean"] = payload.get("results", {}).get("random_masked", {}).get("return_mean")
    return thresholds
