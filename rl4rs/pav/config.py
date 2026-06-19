import os
from dataclasses import dataclass, field


def _capacity_units(capacity):
    if capacity == "small":
        return [128]
    if capacity == "large":
        return [512, 256]
    return [256, 128]


@dataclass
class PAVConfig:
    env: str = "SlateRecEnv-v0"
    trial_name: str = "a_all"
    gamma: float = 1.0
    k: int = None
    alpha: float = 0.1
    clip_c: float = 3.0
    action_size: int = 284
    batch_size: int = 256
    reward_epochs: int = 5
    verifier_epochs: int = 5
    learning_rate: float = 1e-3
    capacity: str = "medium"
    reward_model_zero: bool = False
    use_verifier: bool = True
    use_clipping: bool = True
    use_raw_progress: bool = False
    verifier_label_mode: str = "sign"
    verifier_margin_frac: float = 0.25
    directional_lambda: float = 0.0
    embed_dim: int = 64
    consistency_beta: float = 0.0
    consistency_epochs: int = 2
    online_use_verifier: bool = True
    prover_kind: str = "logging"
    prover_bo_k: int = 3
    prover_artifact_path: str = ""
    use_simulator_q: bool = False
    use_trajectory_q_avg: bool = True
    use_hybrid_mc: bool = True
    hybrid_mc_fraction: float = 0.2
    n_mc: int = 8
    n_cov: int = -1
    mc_seed: int = 0
    max_mc_states: int = 5000
    mc_max_workers: int = 4
    mc_sim_batch_size: int = 1
    mc_sim_use_cpu: bool = False
    mc_progress_every: int = 25
    reward_target: str = "q_mu"
    verifier_output_mode: str = "q_regression"
    normalize_contribution: bool = False
    monitor_every: int = 500
    distinguishability_floor: float = 0.05
    alpha_decay_enabled: bool = True
    alpha_decay_rate: float = 0.1
    max_train_samples: int = None
    device: str = None
    dataset_dir: str = None
    output_dir: str = None
    suffix: str = "pav"
    hidden_units: list = field(default_factory=lambda: [256, 128])

    @classmethod
    def from_dict(cls, values):
        data = dict(values or {})
        valid_keys = set(cls.__dataclass_fields__.keys())
        data = {key: value for key, value in data.items() if key in valid_keys}
        if data.get("k") is None:
            data["k"] = 5 if data.get("env") == "SeqSlateRecEnv-v0" else 3
        if "hidden_units" not in data:
            data["hidden_units"] = _capacity_units(data.get("capacity", "medium"))
        if data.get("dataset_dir") is None:
            data["dataset_dir"] = os.environ.get("rl4rs_dataset_dir", "../dataset")
        if data.get("output_dir") is None:
            data["output_dir"] = os.environ.get("rl4rs_output_dir", "../output")
        return cls(**data)

    @property
    def dataset_name(self):
        return "{}_{}.h5".format(self.env, self.trial_name)

    @property
    def raw_dataset_path(self):
        return os.path.join(self.dataset_dir, self.dataset_name)

    @property
    def shaped_dataset_name(self):
        return "{}_{}_{}.h5".format(self.env, self.trial_name, self.suffix)

    @property
    def shaped_dataset_path(self):
        return os.path.join(self.dataset_dir, self.shaped_dataset_name)

    @property
    def pav_output_dir(self):
        return os.path.join(self.output_dir, "pav")

    @property
    def artifact_prefix(self):
        safe_env = self.env.replace("-", "_")
        return "{}_{}_{}".format(safe_env, self.trial_name, self.suffix)

    @property
    def reward_model_path(self):
        return os.path.join(self.pav_output_dir, "Reward_{}.pt".format(self.artifact_prefix))

    @property
    def verifier_path(self):
        return os.path.join(self.pav_output_dir, "Verifier_{}.pt".format(self.artifact_prefix))

    @property
    def stats_path(self):
        return os.path.join(self.pav_output_dir, "stats_{}.json".format(self.artifact_prefix))
