from rl4rs.pav.config import PAVConfig
from rl4rs.pav.progress import compute_k_step_progress, shape_rewards, verifier_labels


def build_pav_dataset(config_or_dict):
    from rl4rs.pav.pipeline import build_pav_dataset as _build_pav_dataset
    return _build_pav_dataset(config_or_dict)

__all__ = [
    "PAVConfig",
    "build_pav_dataset",
    "compute_k_step_progress",
    "shape_rewards",
    "verifier_labels",
]
