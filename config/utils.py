import json
from typing import List


def load_config(folder_path) -> dict:
    """Load multiple configuration files and combine them into one dictionary."""
    combined_config = {}

    file_paths = ['climate_config.json', 'crop_config.json', 'simulation_config.json']

    for file_path in file_paths:
        with open(folder_path + file_path, 'r') as file:
            config = json.load(file)
            combined_config.update(config)  # This overwrites any existing keys

    return combined_config


def get_attribute(data, key):
    """Get attribute from dictionary. Raise KeyError if not found."""
    if key in data:
        return data[key]
    raise KeyError(f"Key '{key}' not found in config file.")