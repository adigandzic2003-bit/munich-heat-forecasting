"""Configuration and reproducible output helpers."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = ("project", "data", "split", "model")
    if not isinstance(config, dict) or any(section not in config for section in required):
        raise ValueError(f"Configuration must contain {required}")
    if config["project"]["horizon_days"] != 1:
        raise ValueError("This teaching implementation supports a one-day horizon")
    if int(config["project"]["reporting_lag_days"]) < 1:
        raise ValueError("At least a one-day reporting lag is required by this design")
    return config


def save_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
