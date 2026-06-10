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
    def __init__(self, observation_dim, action_size, hidden_units, action_emb_size=32):
        super(Verifier, self).__init__()
        self.action_emb = nn.Embedding(action_size, action_emb_size)
        self.model = MLP(observation_dim + action_emb_size, hidden_units, 1)

    def forward(self, observations, actions):
        actions = actions.long().view(-1)
        action_emb = self.action_emb(actions)
        x = torch.cat([observations, action_emb], dim=-1)
        return self.model(x).squeeze(-1)


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


def load_verifier(path, observation_dim, action_size, hidden_units, device):
    model = Verifier(observation_dim, action_size, hidden_units).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("metadata", {})
