"""Consistent per-target-station temperature forecasts without test-based tuning."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .baselines import SeasonalClimatology, persistence_predict
from .features import feature_columns, make_feature_table
from .models import chronological_splits, regression_model


def compare_target_stations(clean: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a fixed, validation-independent RF for each target station.

    Uses a fixed first candidate from YAML, never chooses by final test.
    Aligns reporting across common held-out calendar dates for fair comparison.
    """
    params = cfg["model"]["regressor_candidates"][0]
    seed = int(cfg["project"]["random_state"])
    frames = []
    for sid in cfg["data"]["stations"]:
        local_cfg = deepcopy(cfg)
        local_cfg["data"]["primary_station"] = sid
        panel, _ = make_feature_table(clean, local_cfg)
        parts = chronological_splits(panel, local_cfg)
        history = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
        test = parts["test"]
        features = feature_columns(history)
        fitted = regression_model(params, seed).fit(history[features], history["target_tmax_c"])
        block = pd.DataFrame({
            "station_id": sid,
            "target_date": pd.to_datetime(test.target_date).to_numpy(),
            "target_tmax_c": test.target_tmax_c.to_numpy(dtype=float),
            "persistence": persistence_predict(test, reporting_lag_days=int(cfg["project"]["reporting_lag_days"])),
            "climatology": SeasonalClimatology().fit(history).predict(test),
            "random_forest": fitted.predict(test[features]),
        })
        frames.append(block)
    if not frames:
        raise ValueError("No stations configured")
    common = set(frames[0].target_date)
    for frame in frames[1:]:
        common &= set(frame.target_date)
    if not common:
        raise ValueError("No common test dates with observed maxima across stations")
    aligned = pd.concat([part.loc[part.target_date.isin(common)] for part in frames], ignore_index=True)
    aligned["season"] = pd.to_datetime(aligned.target_date).dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"})
    rows = []
    for (sid, season), part in aligned.groupby(["station_id", "season"], sort=True):
        for name in ("persistence", "climatology", "random_forest"):
            y, z = part.target_tmax_c.to_numpy(), part[name].to_numpy()
            rows.append({"station_id": sid, "season": season, "method": name, "n_days": len(part),
                         "mae_c": float(mean_absolute_error(y, z)),
                         "rmse_c": float(np.sqrt(mean_squared_error(y, z))),
                         "hot_day_mae_c": float(np.mean(np.abs(y[y>=cfg['project']['hot_threshold_c']] - z[y>=cfg['project']['hot_threshold_c']]))) if (y>=cfg['project']['hot_threshold_c']).any() else np.nan})
    return aligned, pd.DataFrame(rows)
