"""
Central project settings loader for the SPEAR project.

Usage from any sub-project directory:
    import sys; sys.path.insert(0, '..')
    from project_settings import settings

    # Access any setting:
    lr = settings["dl"]["training"]["learning_rate"]
    data_path = settings["paths"]["dataset_file"]

    # Or use the helper:
    from project_settings import get
    lr = get("dl.training.learning_rate")
"""

import json
import os


def _find_config():
    """Find project_config.json by searching current dir and parent dirs."""
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),  # Same dir as this module
        os.getcwd(),                                   # Current working directory
        os.path.join(os.getcwd(), ".."),               # Parent directory
        os.path.join(os.getcwd(), "..", ".."),          # Grandparent directory
    ]

    for d in search_dirs:
        candidate = os.path.join(d, "project_config.json")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "project_config.json not found. Searched: " + ", ".join(search_dirs)
    )


def load_settings(config_path=None):
    """Load project settings from JSON config file."""
    if config_path is None:
        config_path = _find_config()
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(dotted_key, default=None, cfg=None):
    """
    Retrieve a nested setting using dot notation.

    Examples:
        get("dl.training.learning_rate")       -> 2e-5
        get("paths.dataset_file")              -> "../../dataset/filtered_data.json"
        get("general.proxy.http", default="")  -> ""

    Args:
        dotted_key: Dot-separated key path.
        default: Value to return if key is not found.
        cfg: Settings dict (uses global `settings` if None).

    Returns:
        The setting value, or `default` if not found.
    """
    if cfg is None:
        cfg = settings

    keys = dotted_key.split(".")
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def apply_env_overrides(cfg):
    """
    Apply environment variable overrides for common settings.

    Supported env vars:
        SPEAR_GPU_ID       -> general.gpu_id
        SPEAR_DEVICE       -> general.device
        SPEAR_RANDOM_SEED  -> general.random_seed
        SPEAR_HTTP_PROXY   -> general.proxy.http
        SPEAR_HTTPS_PROXY  -> general.proxy.https
    """
    env_map = {
        "SPEAR_GPU_ID": ("general", "gpu_id"),
        "SPEAR_DEVICE": ("general", "device"),
        "SPEAR_HTTP_PROXY": ("general", "proxy", "http"),
        "SPEAR_HTTPS_PROXY": ("general", "proxy", "https"),
    }

    for env_var, key_path in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            current = cfg
            for key in key_path[:-1]:
                current = current[key]
            current[key_path[-1]] = value

    seed_str = os.environ.get("SPEAR_RANDOM_SEED")
    if seed_str is not None:
        cfg["general"]["random_seed"] = int(seed_str)

    return cfg


# Load settings at import time
settings = load_settings()
settings = apply_env_overrides(settings)
