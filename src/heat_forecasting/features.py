"""Leakage-safe feature table anchored to each daily forecast origin."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_feature_table(observations: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    data_cfg = cfg["data"]
    project_cfg = cfg["project"]
    lag = int(project_cfg["reporting_lag_days"])
    horizon = int(project_cfg["horizon_days"])
    primary = data_cfg["primary_station"]
    ids = list(data_cfg["stations"])
    if primary not in ids:
        raise ValueError("Primary station not in the configured station list")
    start = pd.Timestamp(data_cfg["start_date"])
    end = pd.Timestamp(data_cfg["end_date"])
    # Padding calendar days is essential: row shifts over missing days would cause leakage.
    index = pd.date_range(start, end, freq="D", name="forecast_origin")
    panel = pd.DataFrame(index=index)
    series_by_station = {}
    for sid in ids:
        rows = observations.loc[observations.station_id == sid, ["date", "tmax_c"]]
        if rows.duplicated("date").any():
            raise ValueError("Quality checks must remove duplicate station/date records")
        series_by_station[sid] = rows.set_index("date")["tmax_c"].reindex(index)
    target_values = series_by_station[primary]
    panel["target_date"] = panel.index + pd.Timedelta(days=horizon)
    panel["target_tmax_c"] = target_values.shift(-horizon)
    panel["hot_event"] = (panel["target_tmax_c"] >= float(project_cfg["hot_threshold_c"])).astype(float)
    panel.loc[panel["target_tmax_c"].isna(), "hot_event"] = np.nan
    # At issue time on D, observation D-1 is the newest assumed accessible daily record.
    primary_lags = sorted({max(lag, int(x)) for x in data_cfg["primary_lags"]})
    for d in primary_lags:
        panel[f"tmax_lag_{d}"] = target_values.shift(d)
    for window in data_cfg["rolling_days"]:
        panel[f"tmax_rolling_mean_{window}"] = target_values.shift(lag).rolling(int(window), min_periods=int(window)).mean()
    for sid in ids:
        if sid == primary:
            continue
        for d in data_cfg["neighbor_lags"]:
            d = max(lag, int(d))
            panel[f"neighbor_{sid}_lag_{d}"] = series_by_station[sid].shift(d)
    doy = panel["target_date"].dt.dayofyear.to_numpy()
    panel["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    panel["day_cos"] = np.cos(2 * np.pi * doy / 365.25)
    panel["target_month"] = panel["target_date"].dt.month.astype(int)
    panel["target_year"] = panel["target_date"].dt.year.astype(int)
    panel = panel.reset_index()
    panel = panel.dropna(subset=["target_tmax_c", f"tmax_lag_{lag}"]).reset_index(drop=True)
    # Data beyond the declared window may be the target, but not supplied; no fake 2026 labels.
    panel = panel.loc[panel["target_date"] <= end].reset_index(drop=True)
    rules = {
        "forecast_origin": "calendar date D, issued at 18:00 Europe/Berlin (retrospective study)",
        "target_date": f"D+{horizon}",
        "latest_allowed_observation": f"D-{lag}",
        "availability_caveat": "Historical daily data do not prove operational release latency; confirm before real-time use",
        "primary_station": primary, "all_station_ids": ids,
        "weather_feature_columns": sorted([x for x in panel if x.startswith("tmax_") or x.startswith("neighbor_")]),
        "station_join": "exact calendar date; never join different years or row offsets",
    }
    return panel, rules


def feature_columns(frame: pd.DataFrame, *, include_neighbors: bool = True) -> list[str]:
    keep = [c for c in frame if c.startswith("tmax_lag_") or c.startswith("tmax_rolling_") or c.startswith("neighbor_") or c in ("day_sin", "day_cos")]
    if not include_neighbors:
        keep = [c for c in keep if not c.startswith("neighbor_")]
    return keep
