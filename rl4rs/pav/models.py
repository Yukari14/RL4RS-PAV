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
    def __init__(self, observation_dim, hidden_units):
        super(RewardModel, self).__init__()
        self.model = MLP(observation_dim, hidden_units, 1)

    def forward(self, observations):
        return self.model(observations).squeeze(-1)


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
    def __call__(self, observations):
        return torch.zeros((observations.shape[0],), device=observations.device)

    def state_dict(self):
        return {}

    def eval(self):
        return self


def save_checkpoint(path, model, metadata):
    torch.save({"model": model.state_dict(), "metadata": metadata}, path)


def load_reward_model(path, observation_dim, hidden_units, device):
    model = RewardModel(observation_dim, hidden_units).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("metadata", {})


def load_verifier(path, observation_dim, action_size, hidden_units, device):
    model = Verifier(observation_dim, action_size, hidden_units).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("metadata", {})
