from rl4rs.pav.config import PAVConfig
from rl4rs.pav.dataset import export_mdpdataset, is_discrete_actions, load_mdpdataset
from rl4rs.pav.trainer import build_pav_signals


def build_pav_dataset(config_or_dict):
    config = config_or_dict
    if not isinstance(config, PAVConfig):
        config = PAVConfig.from_dict(config_or_dict)

    dataset = load_mdpdataset(config.raw_dataset_path)
    signals = build_pav_signals(dataset, config)
    flat = signals["flat"]
    export_mdpdataset(
        flat["observations"],
        flat["actions"],
        signals["shaped_rewards"],
        flat["terminals"],
        config.shaped_dataset_path,
        discrete_action=is_discrete_actions(flat["actions"]),
    )
    print("PAV shaped dataset saved to {}".format(config.shaped_dataset_path))
    return config.shaped_dataset_path, signals["stats"]
