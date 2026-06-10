#!/usr/bin/env python3
"""Verify incremental online PAV matches full-episode prefix replay."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from rl4rs.online.config import default_pav_config
from rl4rs.pav.online import (
    _EpisodeBuffer,
    apply_pav_to_episode,
    compute_step_shaped_reward,
    load_pav_artifacts,
    shape_latest_steps_batched,
)


def _random_prefix(length, obs_dim=266):
    rng = np.random.RandomState(0)
    prefix = []
    for _ in range(length):
        prefix.append({
            "obs": rng.randn(obs_dim).astype(np.float32),
            "action": int(rng.randint(0, 284)),
            "raw_reward": float(rng.rand() * 10.0),
        })
    return prefix


def main():
    output_dir = os.environ.get("rl4rs_output_dir", os.path.join(ROOT, "output"))
    dataset_dir = os.environ.get("rl4rs_dataset_dir", os.path.join(ROOT, "dataset"))
    pav_config = default_pav_config(
        output_dir, dataset_dir, trial_name="a_50k_logged", suffix="pav_v2"
    )
    artifacts = load_pav_artifacts(pav_config)

    max_diff = 0.0
    for length in range(1, 10):
        prefix = _random_prefix(length)
        incremental = compute_step_shaped_reward(prefix, artifacts)
        offline_last = float(apply_pav_to_episode(prefix, artifacts)[-1])

        buf = _EpisodeBuffer()
        for t in prefix:
            buf.append(t["obs"], t["action"], t["raw_reward"])
        wrapped = float(
            shape_latest_steps_batched(
                None, None, [prefix[-1]["raw_reward"]], [length - 1], artifacts, [buf]
            )[0]
        )

        for val, name in (
            (incremental, "incremental_api"),
            (wrapped, "wrapper_buffer"),
            (offline_last, "episode_replay"),
        ):
            diff = abs(incremental - val)
            max_diff = max(max_diff, diff)
            assert diff < 1e-4, (length, name, incremental, val, diff)

    print("PAV online inference equivalence test passed (max_diff={:.2e})".format(max_diff))
    return 0


if __name__ == "__main__":
    sys.exit(main())
