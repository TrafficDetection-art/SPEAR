"""
Output directory management module.
"""

import sys; sys.path.insert(0, '..')
from project_settings import get as pget

import os
from datetime import datetime
from config import config


def create_output_directory():
    """
    Create the output directory structure based on model names and timestamp.

    Returns:
        str: Path to the created output directory.
    """
    output_base_dir = pget("spear.output_base_dir", "outputs")

    attack_model = config["attack_model"]["model"]
    defense_model = config["defense_model"]["model"]

    if attack_model == defense_model:
        model_name = f"agent-{attack_model}"
    else:
        model_name = f"agent-attack_{attack_model}_vs_defense_{defense_model}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, model_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir
