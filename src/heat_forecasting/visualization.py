"""Offline figures for audit, evaluation and teaching defense."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figures(prediction: pd.DataFrame, regression: pd.DataFrame, calibration: pd.DataFrame, outdir: str | Path, cfg: dict) -> list[str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    frame = prediction.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    view = frame.loc[frame["target_date"].dt.year == frame["target_date"].dt.year.min()].head(180)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(view.target_date, view.target_tmax_c, label="Observed maximum", linewidth=1.6)
    ax.plot(view.target_date, view.random_forest, label="Random forest forecast", alpha=0.8)
    ax.axhline(float(cfg["project"]["hot_threshold_c"]), color="red", linestyle="--", label="Hot-day threshold")
    ax.set(ylabel="Daily maximum (°C)", title="Held-out forecast timeline (first 180 days)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out / "test_timeline.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    overall = regression.loc[regression.segment == "overall"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(overall.model, overall.mae_c)
    ax.set(ylabel="MAE (°C)", title="Held-out temperature error by method")
    ax.tick_params(axis="x", labelrotation=17)
    fig.tight_layout()
    path = out / "model_comparison.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")
    ax.scatter(calibration.mean_predicted, calibration.observed_frequency, s=20 + np.sqrt(calibration.n) * 8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed event frequency", title="Hot-day probability calibration")
    ax.legend()
    fig.tight_layout()
    path = out / "calibration.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    seasons = regression[(regression.model.isin(["persistence", "random_forest"])) & (regression.segment != "overall")]
    wide = seasons.pivot(index="segment", columns="model", values="mae_c")
    wide.plot(kind="bar", ax=ax)
    ax.set(xlabel="Season", ylabel="MAE (°C)", title="Error by season")
    fig.tight_layout()
    path = out / "seasonal_error.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    return paths
