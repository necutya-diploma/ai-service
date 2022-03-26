import json


def read_configs(config_path: str) -> dict:
    config = {}

    with open(config_path, 'r') as config_file:
        config.update(json.load(config_file))

    return config
