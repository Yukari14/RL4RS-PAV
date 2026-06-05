import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl4rs.pav.config import PAVConfig
from rl4rs.pav.diagnostics import (
    write_ablation_matrix,
    write_experiment_matrix,
    write_step_diagnostics,
)


stage = sys.argv[1] if len(sys.argv) >= 2 else "shape_dataset"
extra_config = eval(sys.argv[2]) if len(sys.argv) >= 3 else {}

base_config = {
    "env": "SlateRecEnv-v0",
    "trial_name": "a_all",
    "action_size": 284,
    "gamma": 1.0,
    "alpha": 0.1,
    "clip_c": 3.0,
    "batch_size": 256,
    "reward_epochs": 5,
    "verifier_epochs": 5,
}
base_config = dict(base_config, **extra_config)
config = PAVConfig.from_dict(base_config)


if stage == "shape_dataset":
    from rl4rs.pav.pipeline import build_pav_dataset

    build_pav_dataset(config)
elif stage == "diagnostics":
    from d3rlpy.dataset import MDPDataset
    from rl4rs.pav.trainer import build_pav_signals

    dataset = MDPDataset.load(config.raw_dataset_path)
    signals = build_pav_signals(dataset, config)
    output_path = os.path.join(
        config.pav_output_dir,
        "diagnostics_{}.csv".format(config.artifact_prefix),
    )
    write_step_diagnostics(signals, output_path)
    print("PAV diagnostics saved to {}".format(output_path))
elif stage == "experiment_matrix":
    output_path = os.path.join(config.pav_output_dir, "experiment_matrix.csv")
    write_experiment_matrix(output_path)
    print("PAV experiment matrix saved to {}".format(output_path))
elif stage == "ablation_matrix":
    output_path = os.path.join(config.pav_output_dir, "ablation_matrix.csv")
    write_ablation_matrix(output_path)
    print("PAV ablation matrix saved to {}".format(output_path))
else:
    raise ValueError("Unknown PAV stage: {}".format(stage))
