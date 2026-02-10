"""
Configuration loading module.

Supports both unified and separated attack/defense model configurations.
"""

import sys; sys.path.insert(0, '..')
from project_settings import get as pget

import json


def load_config(config_path=None):
    """
    Load configuration file with support for separate attack/defense model configs.

    Config file format (new):
    {
        "attack_model": {
            "api_key": "...",
            "api_base_url": "...",
            "model": "gpt-4o"
        },
        "defense_model": {
            "api_key": "...",
            "api_base_url": "...",
            "model": "gpt-4o-mini"
        }
    }

    Legacy format (also supported):
    {
        "api_key": "...",
        "api_base_url": "...",
        "model": "gpt-4o"
    }
    """
    if config_path is None:
        config_path = pget("spear.config_path", "./config.json")
    with open(config_path, "r") as config_file:
        config_data = json.load(config_file)

    # Check if using new format (separate attack/defense)
    if "attack_model" in config_data and "defense_model" in config_data:
        print("Detected separate attack/defense configuration mode")
        print(f"  Attack model: {config_data['attack_model'].get('model', 'unknown')}")
        print(f"  Defense model: {config_data['defense_model'].get('model', 'unknown')}")
        return config_data
    else:
        # Backward compatible: attack and defense share the same config
        print("Using unified configuration mode (shared model for attack/defense)")
        print(f"  Model: {config_data.get('model', 'unknown')}")
        return {
            "attack_model": config_data,
            "defense_model": config_data,
        }


# Global configuration loaded at import time
config = load_config()
