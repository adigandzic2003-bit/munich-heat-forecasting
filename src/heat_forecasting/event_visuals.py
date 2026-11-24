"""Ex-post illustrations of extreme weather episodes, never used for tuning."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heat_episode(predictions: pd.DataFrame, episodes: pd.DataFrame, destination: str | Path, *, days_margin: int = 3) -> Path | None:
    if episodes.empty:
        return None
    # Explicitly ex-post illustrative selection; no model choice based on this plot.
    highest = episodes.sort_values(["peak_tmax_c", "length_days"], ascending=False).iloc[0]
    start, end = pd.Timestamp(highest.start_date), pd.Timestamp(highest.end_date)
    pred = predictions.copy()
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    part = pred.loc[pred.target_date.between(start - pd.Timedelta(days=days_margin), end + pd.Timedelta(days=days_margin))].sort_values("target_date")
    if part.empty:
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(part.target_date, part.target_tmax_c, marker="o", label="Observed maximum")
    ax.plot(part.target_date, part.random_forest, marker=".", label="Predicted maximum")
    ax.axhline(30, color="gray", linestyle=":", label="30°C threshold")
    ax.axvspan(start, end, color="silver", alpha=.25, label="Observed episode")
    ax.set(xlabel="Target date", ylabel="Temperature (°C)", title="Selected observed heat episode (ex-post illustration)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(destination,dpi=150);plt.close(fig)
    return destination


def extreme_day_diagnostics(predictions: pd.DataFrame, *, threshold_c: float = 30.0) -> pd.DataFrame:
    work = predictions.copy()
    required = {"target_date", "target_tmax_c", "random_forest", "heat_alert"}
    if required - set(work):
        raise ValueError(f"Missing predictions fields: {required-set(work)}")
    work["error_c"] = work.random_forest - work.target_tmax_c
    work["abs_error_c"] = work.error_c.abs()
    work["actual_hot"] = work.target_tmax_c >= threshold_c
    work["missed_alert"] = work.actual_hot & work.heat_alert.eq(0)
    work["false_alarm"] = ~work.actual_hot & work.heat_alert.eq(1)
    return work.loc[work.actual_hot | work.false_alarm, ["target_date", "target_tmax_c", "random_forest", "error_c", "abs_error_c", "actual_hot", "missed_alert", "false_alarm"]].sort_values(["missed_alert", "abs_error_c"], ascending=False).reset_index(drop=True)
