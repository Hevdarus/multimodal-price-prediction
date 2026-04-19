from __future__ import annotations

import re
from pathlib import Path


def load_experiments_config(config_path: str | Path) -> dict[str, dict]:
    """
    Load experiment configurations from a plain text file.

    Expected block format:

    experiment_name-et: "text_distilbert_lr1e5_len64_ep3"
    lr: 1e-5
    max_length: 64
    epochs: 3
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8").strip()

    blocks = re.split(r"\n\s*\n", text)
    experiments = {}

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        config = {}

        for line in lines:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key in {"experiment_name", "experiment_name-et"}:
                config["experiment_name"] = value.strip('"').strip("'")
            elif key == "lr":
                config["lr"] = float(value)
            elif key == "max_length":
                config["max_length"] = int(value)
            elif key == "epochs":
                config["epochs"] = int(value)

        required_keys = {"experiment_name", "lr", "max_length", "epochs"}
        missing = required_keys - set(config.keys())
        if missing:
            raise ValueError(
                f"Incomplete experiment block. Missing keys: {missing}. Block:\n{block}"
            )

        experiments[config["experiment_name"]] = config

    if not experiments:
        raise ValueError("No valid experiment configuration found.")

    return experiments


def get_experiment_config(config_path: str | Path, experiment_name: str) -> dict:
    """
    Return the config dict for a specific experiment name.
    """
    experiments = load_experiments_config(config_path)

    if experiment_name not in experiments:
        available = list(experiments.keys())
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Available experiments: {available}"
        )

    return experiments[experiment_name]