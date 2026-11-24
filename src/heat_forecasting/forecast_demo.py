"""An auditable retrospective example: forecast D+1 using data no newer than D-lag."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def example_forecast(predictions: pd.DataFrame, cfg: dict, *, target_date: str | None = None) -> dict:
    frame = predictions.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    if frame.empty:
        raise ValueError("No test predictions")
    if target_date is None:
        # A predeclared rule, not a manual choice after looking at model accuracy.
        # Pick the chronological median held-out target to show how a single issue works.
        sample = frame.sort_values("target_date").iloc[len(frame) // 2]
    else:
        matching = frame.loc[frame.target_date.eq(pd.Timestamp(target_date))]
        if len(matching) != 1:
            raise ValueError("Requested target date must appear exactly once in held-out predictions")
        sample = matching.iloc[0]
    lag = int(cfg["project"]["reporting_lag_days"])
    horizon = int(cfg["project"]["horizon_days"])
    origin, target = pd.Timestamp(sample.forecast_origin), pd.Timestamp(sample.target_date)
    if target != origin + pd.Timedelta(days=horizon):
        raise AssertionError("Forecast target/horizon mismatch")
    cutoff = origin - pd.Timedelta(days=lag)
    return {
        "status": "RETROSPECTIVE EXAMPLE; measured outcome disclosed ex-post",
        "target_station": cfg["data"]["primary_station"],
        "forecast_issued_date": origin.date().isoformat(),
        "forecast_issued_hour_local": int(cfg["project"]["issue_hour_local"]),
        "latest_allowed_observation_date": cutoff.date().isoformat(),
        "forecast_target_date": target.date().isoformat(),
        "horizon_days": horizon,
        "predicted_tmax_c": float(sample.random_forest),
        "persistence_tmax_c": float(sample.persistence),
        "observed_tmax_c_ex_post": float(sample.target_tmax_c),
        "heat_probability": float(sample.heat_probability),
        "heat_alert": bool(sample.heat_alert),
        "note": "The issue-time cutoff is an explicit scenario assumption, not verified DWD operational publication latency.",
    }


def plot_forecast_example(predictions: pd.DataFrame, example: dict, destination: str | Path, *, days_each_side: int = 5) -> Path:
    date = pd.Timestamp(example["forecast_target_date"])
    frame = predictions.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame = frame.loc[frame.target_date.between(date - pd.Timedelta(days=days_each_side), date + pd.Timedelta(days=days_each_side))].sort_values("target_date")
    if frame.empty:
        raise ValueError("No data for example plot")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame.target_date, frame.target_tmax_c, marker="o", label="Observed ex-post")
    ax.plot(frame.target_date, frame.random_forest, marker=".", label="Random forest")
    ax.axhline(30, linestyle=":", color="grey", label="30°C hot-day threshold")
    ax.axvline(date, linestyle="--", color="black", label="Example forecast target")
    ax.set(xlabel="Target date", ylabel="Max temperature (°C)", title="Retrospective issued forecast and neighboring days")
    ax.legend(fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)
    return destination
