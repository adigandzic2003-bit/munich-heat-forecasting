"""Descriptive, ex-post climatology. Never select models using held-out EDA."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def descriptive_weather_tables(observations: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily Munich coverage, annual temperature summaries, and monthly climatology.

    Includes the final-test period for DESCRIPTIVE reporting only. The model
    fitting, validation and model selection routines never read these tables.
    """
    sid = str(cfg["data"]["primary_station"])
    work = observations.loc[observations.station_id.astype(str).str.zfill(5) == sid, ["date", "tmax_c"]].copy()
    if work.empty:
        raise ValueError(f"No observed temperatures for primary station {sid}")
    work["date"] = pd.to_datetime(work["date"])
    if work.date.duplicated().any():
        raise ValueError("Descriptive climate series needs exactly one row per calendar date")
    start, end = pd.Timestamp(cfg["data"]["start_date"]), pd.Timestamp(cfg["data"]["end_date"])
    if end < start:
        raise ValueError("Invalid date range")
    full_calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    full_calendar = full_calendar.merge(work, on="date", how="left", validate="one_to_one", indicator=True)
    full_calendar["year"] = full_calendar.date.dt.year
    full_calendar["month"] = full_calendar.date.dt.month
    full_calendar["observed"] = full_calendar["_merge"] == "both"
    full_calendar["valid"] = full_calendar.tmax_c.notna()
    threshold = float(cfg["project"]["hot_threshold_c"])
    full_calendar["hot_day"] = full_calendar.tmax_c.ge(threshold)

    annual = full_calendar.groupby("year", as_index=False).agg(
        calendar_days=("date", "size"), recorded_days=("observed", "sum"),
        valid_temperature_days=("valid", "sum"),
        mean_tmax_c=("tmax_c", "mean"), p95_tmax_c=("tmax_c", lambda x: x.quantile(0.95)),
        maximum_tmax_c=("tmax_c", "max"), observed_hot_days=("hot_day", "sum"),
    )
    annual["coverage_fraction"] = annual.valid_temperature_days / annual.calendar_days
    annual["missing_calendar_days"] = annual.calendar_days - annual.recorded_days
    annual["hot_day_definition_c"] = threshold
    monthly = full_calendar.groupby("month", as_index=False).agg(
        valid_temperature_days=("valid", "sum"), mean_tmax_c=("tmax_c", "mean"),
        p10_tmax_c=("tmax_c", lambda x: x.quantile(0.1)),
        p90_tmax_c=("tmax_c", lambda x: x.quantile(0.9)),
        total_hot_days=("hot_day", "sum"),
    )
    monthly["hot_day_definition_c"] = threshold
    return annual, monthly


def plot_descriptive_weather(annual: pd.DataFrame, monthly: pd.DataFrame, folder: str | Path) -> list[str]:
    """Produce plain, readable descriptive figures (not model-selection evidence)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(annual.year, annual.mean_tmax_c, marker=".")
    ax.set(xlabel="Year", ylabel="Mean daily maximum (°C)", title="Munich historical temperatures (descriptive)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = folder / "eda_annual_mean_tmax.png"
    fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(annual.year, annual.observed_hot_days)
    ax.set(xlabel="Year", ylabel="Observed hot days", title="Days at or above the defined hot threshold")
    fig.tight_layout()
    path = folder / "eda_annual_hot_days.png"
    fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.month, monthly.mean_tmax_c, marker="o", label="Monthly mean")
    ax.fill_between(monthly.month.to_numpy(), monthly.p10_tmax_c.to_numpy(), monthly.p90_tmax_c.to_numpy(), alpha=0.18, label="10th–90th percentiles")
    ax.set(xlim=(1, 12), xticks=list(range(1, 13)), xlabel="Calendar month", ylabel="Daily maximum (°C)", title="Seasonal temperature pattern")
    ax.legend()
    fig.tight_layout()
    path = folder / "eda_monthly_climatology.png"
    fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    return paths
