import numpy as np
import importlib.util
import os

module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rl4rs", "pav", "progress.py")
)
spec = importlib.util.spec_from_file_location("pav_progress", module_path)
pav_progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pav_progress)

compute_k_step_progress = pav_progress.compute_k_step_progress
shape_rewards = pav_progress.shape_rewards
verifier_labels = pav_progress.verifier_labels


flat = {
    "rewards": np.asarray([0.0, 0.0, 10.0, 0.0, 5.0], dtype="float32"),
    "episode_ids": np.asarray([0, 0, 0, 1, 1], dtype="int64"),
    "step_ids": np.asarray([0, 1, 2, 0, 1], dtype="int64"),
    "returns": np.asarray([10.0, 10.0, 10.0, 5.0, 5.0], dtype="float32"),
}
values = np.asarray([3.0, 4.0, 8.0, 1.0, 2.0], dtype="float32")

progress = compute_k_step_progress(flat, values, k=2, gamma=1.0)
expected = np.asarray([5.0, 6.0, 2.0, 4.0, 3.0], dtype="float32")
assert np.allclose(progress, expected), (progress, expected)

labels, progress_baseline, return_baseline = verifier_labels(
    progress, flat["returns"], flat["step_ids"]
)
assert labels.shape == progress.shape
assert progress_baseline.shape == progress.shape
assert return_baseline.shape == progress.shape

shaped, normalized, stats = shape_rewards(
    flat["rewards"], progress, flat["step_ids"], alpha=0.1, clip_c=1.0
)
assert shaped.shape == flat["rewards"].shape
assert normalized.max() <= 1.0
assert normalized.min() >= -1.0
assert "0" in stats and "1" in stats and "2" in stats

print("PAV progress smoke test passed")
