"""Weather forecast and rare-event evaluation, plus year-block uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, precision_score, recall_score,
)

SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
REGRESSION_COLUMNS = ("persistence", "climatology", "random_forest", "ablation_no_neighbors")


def regression_scores(prediction: pd.DataFrame, hot_threshold_c: float = 30.0) -> pd.DataFrame:
    records = []
    frame = prediction.copy()
    frame["season"] = pd.to_datetime(frame["target_date"]).dt.month.map(SEASONS)
    for name in REGRESSION_COLUMNS:
        if name not in frame:
            continue
        for group_name, subset in [("overall", frame), *(list(frame.groupby("season", sort=True)))]:
            y = subset["target_tmax_c"]
            estimate = subset[name]
            hot = y >= hot_threshold_c
            records.append({"model": name, "segment": group_name, "n_days": len(subset),
                            "mae_c": float(mean_absolute_error(y, estimate)),
                            "rmse_c": float(np.sqrt(mean_squared_error(y, estimate))),
                            "hot_days_diagnostic_mae_c": float(mean_absolute_error(y[hot], estimate[hot])) if hot.any() else np.nan})
    return pd.DataFrame(records)


def event_scores(prediction: pd.DataFrame, cfg: dict) -> dict:
    actual = prediction["hot_event"].astype(int).to_numpy()
    alerts = prediction["heat_alert"].astype(int).to_numpy()
    prob = prediction["heat_probability"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(actual, alerts, labels=[0, 1]).ravel()
    threshold = float(cfg["project"]["hot_threshold_c"])
    target_alert = (prediction["random_forest"].to_numpy() >= threshold).astype(int)
    return {
        "n_days": int(len(actual)), "observed_hot_days": int(actual.sum()),
        "hot_day_definition_c": threshold,
        "true_positives": int(tp), "false_positives": int(fp), "false_negatives": int(fn), "true_negatives": int(tn),
        "precision": float(precision_score(actual, alerts, zero_division=0)),
        "recall": float(recall_score(actual, alerts, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0,
        "brier_score": float(brier_score_loss(actual, prob)),
        "average_precision": float(average_precision_score(actual, prob)) if actual.sum() else None,
        "temperature_model_event_recall": float(recall_score(actual, target_alert, zero_division=0)),
    }


def calibration_table(prediction: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    frame = prediction.copy()
    frame["probability_bin"] = pd.cut(frame["heat_probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = frame.groupby("probability_bin", observed=False)
    return grouped.agg(n=("hot_event", "size"), mean_predicted=("heat_probability", "mean"), observed_frequency=("hot_event", "mean")).dropna().reset_index().assign(probability_bin=lambda df: df.probability_bin.astype(str))


def year_block_mae_interval(prediction: pd.DataFrame, cfg: dict) -> dict:
    """Resample complete calendar years instead of treating consecutive hot days as independent."""
    rng = np.random.default_rng(int(cfg["project"]["random_state"]))
    frame = prediction.copy()
    frame["year"] = pd.to_datetime(frame["target_date"]).dt.year
    groups = [grp for _, grp in frame.groupby("year")]
    if len(groups) < 2:
        return {"note": "At least two test years required for block resampling"}
    draws = []
    for _ in range(int(cfg["model"]["bootstrap_repetitions"])):
        chosen = rng.integers(0, len(groups), len(groups))
        errors = np.concatenate([(groups[i]["target_tmax_c"] - groups[i]["random_forest"]).abs().to_numpy() for i in chosen])
        draws.append(float(errors.mean()))
    return {"method": "calendar-year block bootstrap", "n_years": len(groups), "repetitions": len(draws),
            "mae_95pct_lower_c": float(np.percentile(draws, 2.5)), "mae_95pct_upper_c": float(np.percentile(draws, 97.5))}
