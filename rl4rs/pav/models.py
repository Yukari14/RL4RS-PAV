import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RewardModel(nn.Module):
    """State value model with optional embedding head for directional progress."""

    def __init__(self, observation_dim, hidden_units, embed_dim=0):
        super(RewardModel, self).__init__()
        self.observation_dim = int(observation_dim)
        self.hidden_units = list(hidden_units)
        self.embed_dim = int(embed_dim)

        trunk_layers = []
        last_dim = observation_dim
        for hidden_dim in hidden_units:
            trunk_layers.append(nn.Linear(last_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.value_head = nn.Linear(last_dim, 1)
        self.embed_head = nn.Linear(last_dim, embed_dim) if embed_dim > 0 else None

    def trunk_forward(self, observations):
        return self.trunk(observations)

    def forward(self, observations):
        hidden = self.trunk_forward(observations)
        return self.value_head(hidden).squeeze(-1)

    def encode(self, observations, normalize=True):
        if self.embed_head is None:
            raise ValueError("RewardModel has no embedding head (embed_dim=0).")
        hidden = self.trunk_forward(observations)
        embeddings = self.embed_head(hidden)
        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings


class Verifier(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_size,
        hidden_units,
        action_emb_size=32,
        output_mode="binary",
    ):
        super(Verifier, self).__init__()
        self.output_mode = str(output_mode)
        self.action_emb = nn.Embedding(action_size, action_emb_size)
        trunk_layers = []
        last_dim = observation_dim + action_emb_size
        for hidden_dim in hidden_units:
            trunk_layers.append(nn.Linear(last_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.gate_head = (
            nn.Linear(last_dim, 1) if self.output_mode in ("binary", "dual") else None
        )
        self.q_head = (
            nn.Linear(last_dim, 1) if self.output_mode in ("q_regression", "dual") else None
        )
        if self.gate_head is None and self.q_head is None:
            self.gate_head = nn.Linear(last_dim, 1)

    def _hidden(self, observations, actions):
        actions = actions.long().view(-1)
        action_emb = self.action_emb(actions)
        x = torch.cat([observations, action_emb], dim=-1)
        return self.trunk(x)

    def forward(self, observations, actions):
        hidden = self._hidden(observations, actions)
        if self.output_mode == "q_regression":
            return self.q_head(hidden).squeeze(-1)
        if self.output_mode == "dual":
            return self.gate_head(hidden).squeeze(-1), self.q_head(hidden).squeeze(-1)
        return self.gate_head(hidden).squeeze(-1)

    def predict_q(self, observations, actions):
        hidden = self._hidden(observations, actions)
        if self.q_head is None:
            raise ValueError("Verifier has no q_head (output_mode={}).".format(self.output_mode))
        return self.q_head(hidden).squeeze(-1)

    def predict_gate_logits(self, observations, actions):
        hidden = self._hidden(observations, actions)
        if self.gate_head is None:
            raise ValueError("Verifier has no gate_head (output_mode={}).".format(self.output_mode))
        return self.gate_head(hidden).squeeze(-1)


class ZeroRewardModel(object):
    def __init__(self, embed_dim=0):
        self.embed_dim = int(embed_dim)

    def __call__(self, observations):
        return torch.zeros((observations.shape[0],), device=observations.device)

    def encode(self, observations, normalize=True):
        if self.embed_dim <= 0:
            raise ValueError("ZeroRewardModel has no embedding head.")
        return torch.zeros(
            (observations.shape[0], self.embed_dim),
            device=observations.device,
        )

    def state_dict(self):
        return {}

    def eval(self):
        return self


def save_checkpoint(path, model, metadata):
    torch.save({"model": model.state_dict(), "metadata": metadata}, path)


def load_reward_model(path, observation_dim, hidden_units, device, embed_dim=0):
    checkpoint = torch.load(path, map_location=device)
    meta = checkpoint.get("metadata", {})
    embed_dim = int(meta.get("embed_dim", embed_dim))
    model = RewardModel(observation_dim, hidden_units, embed_dim=embed_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, meta


def _migrate_legacy_verifier_state(state):
    if not any(key.startswith("model.") for key in state):
        return state
    migrated = {}
    for key, value in state.items():
        if key.startswith("action_emb."):
            migrated[key] = value
        elif key.startswith("model."):
            suffix = key[len("model."):]
            parts = suffix.split(".")
            layer_idx = int(parts[0])
            tail = ".".join(parts[1:])
            if layer_idx >= 4:
                migrated["gate_head.{}".format(tail)] = value
            else:
                migrated["trunk.{}.{}".format(layer_idx, tail)] = value
    return migrated


def load_verifier(path, observation_dim, action_size, hidden_units, device, output_mode="binary"):
    checkpoint = torch.load(path, map_location=device)
    meta = checkpoint.get("metadata", {})
    output_mode = meta.get("verifier_output_mode", output_mode)
    model = Verifier(
        observation_dim,
        action_size,
        hidden_units,
        output_mode=output_mode,
    ).to(device)
    state = _migrate_legacy_verifier_state(checkpoint["model"])
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, meta
