import gym
import numpy as np

from rl4rs.env.slate import SlateRecEnv, SlateState

OBS_FEATURE_DIM = 256
MASK_SIZE = 10
PAGE_ITEMS = 9


def load_slate_masks(iteminfo_file, action_size=284):
    item_info = open(iteminfo_file, "r").read().split("\n")[1:]
    item_info = [x.split(" ") for x in item_info if x.strip()]
    special_items = [
        int(itemid)
        for (itemid, _item_vec, _price, _location, is_special) in item_info
        if int(is_special) == 2
    ]
    location_mask = np.zeros((4, action_size), dtype=np.int32)
    location_mask[0, 1:40] = 1
    location_mask[1, 40:148] = 1
    location_mask[2, 148:] = 1
    location_mask[3, 0] = 1
    return location_mask, special_items


def attach_slate_masks(config):
    location_mask, special_items = SlateState.get_mask_from_file(
        config["iteminfo_file"], config["action_size"]
    )
    config["location_mask"] = location_mask
    config["special_items"] = special_items
    config["mask_size"] = MASK_SIZE
    config["page_items"] = PAGE_ITEMS
    return config


def make_slate_env(config):
    if "location_mask" not in config:
        attach_slate_masks(config)
    sim = SlateRecEnv(config, state_cls=SlateState)
    return gym.make("SlateRecEnv-v0", recsim=sim)


def obs_list(obs):
    if isinstance(obs, dict):
        return [obs]
    if isinstance(obs, np.ndarray):
        if obs.ndim == 1:
            return [obs]
        return list(obs)
    return obs


def obs_vector(obs):
    """266-d flat obs (256 simulator + prev actions + step), matches offline PAV."""
    if isinstance(obs, dict):
        if "obs" in obs:
            raise ValueError(
                "rllib dict obs has no prev-action tail; use pav_obs_vector_from_env(env, index)"
            )
        raise KeyError("unexpected obs dict format")
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape[-1] != OBS_FEATURE_DIM + MASK_SIZE:
        raise ValueError("expected obs dim {}, got {}".format(OBS_FEATURE_DIM + MASK_SIZE, arr.shape[-1]))
    return arr


def pav_obs_vector_from_env(env, index=0):
    """Build 266-d PAV features from SlateRecEnv regardless of rllib/d3rl mask mode."""
    inner = getattr(env, "env", env)
    sim = inner.sim
    samples = inner.samples
    state = samples.state
    raw_state = state["state"] if isinstance(state, dict) else state
    feat, _ = sim.FeatureUtil.feature_extraction(raw_state)
    with sim.sess.as_default():
        with sim.sess.graph.as_default():
            obs = sim.obs_layer(feat)
    masked_actions = samples.prev_actions
    cur_steps = np.full((samples.batch_size, 1), samples.cur_steps, dtype=np.float32)
    vec = np.concatenate([obs, masked_actions, cur_steps], axis=-1)
    return np.asarray(vec[index], dtype=np.float32)


def pav_obs_for_step(env, gym_obs, index=0):
    """Convert a pre-step gym observation to 266-d PAV input."""
    if isinstance(gym_obs, dict) and "obs" in gym_obs:
        return pav_obs_vector_from_env(env, index=index)
    return obs_vector(gym_obs)


def batch_pav_obs_vectors_from_env(env, gym_obs=None):
    """Build (batch_size, 266) PAV features in one simulator forward when possible."""
    inner = getattr(env, "env", env)
    items = obs_list(gym_obs if gym_obs is not None else inner.state)
    if not items:
        return np.zeros((0, OBS_FEATURE_DIM + MASK_SIZE), dtype=np.float32)
    if isinstance(items[0], dict) and "obs" in items[0]:
        sim = inner.sim
        samples = inner.samples
        state = samples.state
        raw_state = state["state"] if isinstance(state, dict) else state
        feat, _ = sim.FeatureUtil.feature_extraction(raw_state)
        with sim.sess.as_default():
            with sim.sess.graph.as_default():
                obs = sim.obs_layer(feat)
        masked_actions = samples.prev_actions
        cur_steps = np.full((samples.batch_size, 1), samples.cur_steps, dtype=np.float32)
        return np.concatenate([obs, masked_actions, cur_steps], axis=-1).astype(np.float32)
    return np.stack([obs_vector(o) for o in items], axis=0).astype(np.float32)


def valid_action_mask(obs, config):
    """Boolean mask over discrete actions (True = valid)."""
    obs = np.asarray(obs_vector(obs), dtype=np.float32)
    if obs.ndim > 1:
        raise NotImplementedError("batched mask not implemented for pilot")
    prev_actions = obs[-MASK_SIZE:-1].astype(int)
    cur_step = int(obs[-1])
    layer = cur_step % PAGE_ITEMS // 3
    mask = config["location_mask"][layer].astype(bool).copy()
    for i in range(MASK_SIZE - 1):
        mask[prev_actions[i]] = False
    if len(np.intersect1d(prev_actions, config["special_items"])) > 0:
        mask[np.asarray(config["special_items"], dtype=int)] = False
    return mask


def action_mask_vector(obs, config):
    return valid_action_mask(obs, config).astype(np.int32)


def sample_masked_actions(obs, rng, config):
    actions = []
    for o in obs_list(obs):
        mask = valid_action_mask(o, config)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("empty action mask")
        actions.append(int(rng.choice(valid)))
    return actions


def mask_stats(obs, config):
    sizes = [int(np.sum(valid_action_mask(o, config))) for o in obs_list(obs)]
    return {
        "min_valid": int(min(sizes)),
        "max_valid": int(max(sizes)),
        "mean_valid": float(np.mean(sizes)),
    }


def as_batch_list(value, batch_size):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = list(value)
        if len(arr) == batch_size:
            return arr
    return [value] * batch_size
