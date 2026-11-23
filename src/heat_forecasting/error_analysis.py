"""Strictly ex-post heat-episode and annual error analyses of untouched test predictions."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def heat_episodes(predictions: pd.DataFrame, *, threshold_c: float = 30.0, min_days: int = 2) -> pd.DataFrame:
    """Find observed consecutive calendar-date heat episodes; score alert detection.

    A missed observation date always breaks an episode; rows are never assumed
    to be consecutive just because they are adjacent in the DataFrame.
    """
    if min_days < 1:
        raise ValueError("min_days must be positive")
    required = {"target_date", "target_tmax_c", "heat_alert", "heat_probability", "random_forest"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}")
    ordered = predictions[list(required)].copy().sort_values("target_date")
    ordered["target_date"] = pd.to_datetime(ordered["target_date"])
    if ordered.target_date.duplicated().any():
        raise ValueError("Episode dates must be unique")
    observed_hot = ordered.target_tmax_c >= threshold_c
    adjacent_day = ordered.target_date.diff().dt.days.eq(1)
    starts = observed_hot & (~observed_hot.shift(fill_value=False) | ~adjacent_day)
    labels = starts.cumsum().where(observed_hot)
    rows = []
    for episode_id, part in ordered.loc[observed_hot].groupby(labels.loc[observed_hot], sort=True):
        if len(part) < min_days:
            continue
        alerts = part["heat_alert"].astype(int)
        rows.append({
            "episode_id": int(episode_id),
            "start_date": part.target_date.min().date().isoformat(),
            "end_date": part.target_date.max().date().isoformat(),
            "length_days": int(len(part)),
            "peak_tmax_c": float(part.target_tmax_c.max()),
            "mean_tmax_c": float(part.target_tmax_c.mean()),
            "any_alert": bool(alerts.any()),
            "alerted_days": int(alerts.sum()),
            "missed_days": int((1 - alerts).sum()),
            "mean_event_probability": float(part.heat_probability.mean()),
            "rf_mae_during_episode_c": float(mean_absolute_error(part.target_tmax_c, part.random_forest)),
        })
    return pd.DataFrame(rows, columns=["episode_id", "start_date", "end_date", "length_days", "peak_tmax_c",
                                       "mean_tmax_c", "any_alert", "alerted_days", "missed_days",
                                       "mean_event_probability", "rf_mae_during_episode_c"])


def annual_test_errors(predictions: pd.DataFrame, *, threshold_c: float = 30.0) -> pd.DataFrame:
    """Observed vs predicted yearly errors. These numbers do not tune the model."""
    needed = {"target_date", "target_tmax_c", "random_forest", "persistence", "heat_alert"}
    if needed - set(predictions):
        raise ValueError(f"Missing columns {needed - set(predictions)}")
    frame = predictions.copy()
    frame["year"] = pd.to_datetime(frame.target_date).dt.year
    records = []
    for year, sample in frame.groupby("year", sort=True):
        true_hot = sample.target_tmax_c.ge(threshold_c)
        predicted_alert = sample.heat_alert.astype(bool)
        records.append({
            "year": int(year), "n_days": int(len(sample)), "observed_hot_days": int(true_hot.sum()),
            "rf_mae_c": float(mean_absolute_error(sample.target_tmax_c, sample.random_forest)),
            "rf_rmse_c": float(np.sqrt(mean_squared_error(sample.target_tmax_c, sample.random_forest))),
            "persistence_mae_c": float(mean_absolute_error(sample.target_tmax_c, sample.persistence)),
            "hot_day_recall": float((predicted_alert & true_hot).sum() / true_hot.sum()) if true_hot.any() else np.nan,
            "false_alerts": int((predicted_alert & ~true_hot).sum()),
            "missed_hot_days": int((~predicted_alert & true_hot).sum()),
        })
    return pd.DataFrame(records)


def plot_annual_errors(annual: pd.DataFrame, folder: str | Path) -> list[str]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if annual.empty:
        return []
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(annual.year, annual.rf_mae_c, marker="o", label="Random forest")
    ax.plot(annual.year, annual.persistence_mae_c, marker="o", label="Persistence")
    ax.set(xlabel="Held-out test year", ylabel="MAE (°C)", title="Temperature errors on held-out years")
    ax.set_xticks(annual.year.to_list())
    ax.legend()
    fig.tight_layout()
    path = folder / "test_error_by_year.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return [str(path)]
