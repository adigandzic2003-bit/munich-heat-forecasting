"""Deterministic synthetic fixture to exercise the full pipeline without DWD/network."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_daily_weather(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    dates = pd.date_range(cfg["data"]["start_date"], cfg["data"]["end_date"], freq="D")
    day = dates.dayofyear.to_numpy()
    seasonal = 15 + 12 * np.sin(2 * np.pi * (day - 106) / 365.25)
    persistent_noise = rng.normal(0, 1.0, len(dates))
    persistent_noise = np.convolve(persistent_noise, np.ones(5) / 5, mode="same") * 8
    pulse = rng.normal(0, 2, len(dates))
    baseline = seasonal + persistent_noise + pulse
    frames = []
    records = []
    for i, (sid, meta) in enumerate(cfg["data"]["stations"].items()):
        temp = baseline + rng.normal(0, 0.9, len(dates)) - i * 0.7
        frames.append(pd.DataFrame({"date": dates, "station_id": sid, "tmax_c": temp, "tmean_c": temp - 5, "tmin_c": temp - 9}))
        records.append({"station_id": sid, **meta, "source_url": "SYNTHETIC: deterministic offline demo", "archive_bytes": 0})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(records)
