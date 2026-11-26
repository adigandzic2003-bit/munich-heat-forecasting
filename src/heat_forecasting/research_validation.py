"""Machine-readable research checks independent of estimator scores."""
from __future__ import annotations

import pandas as pd


def audit_research_design(parts: dict[str, pd.DataFrame], predictions: pd.DataFrame, cfg: dict, *, feature_table: pd.DataFrame) -> dict:
    required = {"train", "validation", "test"}
    if required - set(parts):
        raise ValueError("Train/validation/test partitions required")
    ordered = {k: pd.to_datetime(parts[k].target_date) for k in ("train", "validation", "test")}
    separation = bool(ordered["train"].max() < ordered["validation"].min() < ordered["test"].min())
    unique = all(not x.duplicated().any() for x in ordered.values())
    prediction_dates = pd.to_datetime(predictions.target_date)
    matches_test = len(prediction_dates) == len(ordered["test"]) and set(prediction_dates) == set(ordered["test"])
    horizon = int(cfg["project"]["horizon_days"])
    lag = int(cfg["project"]["reporting_lag_days"])
    actual_horizon = (pd.to_datetime(feature_table.target_date) - pd.to_datetime(feature_table.forecast_origin)).dt.days
    horizon_consistent = bool(actual_horizon.eq(horizon).all())
    available_cutoff_before_target = bool(lag >= 1 and horizon >= 1)
    pred_cols = {"random_forest", "ablation_no_neighbors", "heat_probability", "heat_alert"}
    complete_predictions = not (pred_cols - set(predictions)) and bool(predictions[list(pred_cols)].notna().all().all())
    metrics = {
        "chronological_disjoint_order": separation,
        "unique_target_dates_in_each_split": unique,
        "prediction_test_dates_exact_match": matches_test,
        "forecast_horizon_matches_config": horizon_consistent,
        "assumed_information_cutoff_predates_target": available_cutoff_before_target,
        "predictions_complete": complete_predictions,
        "n_train": len(parts["train"]), "n_validation": len(parts["validation"]), "n_test": len(parts["test"]),
        "warning": "Time cutoff is hypothetical and not proof of DWD release latency.",
    }
    if not all([separation, unique, matches_test, horizon_consistent, available_cutoff_before_target, complete_predictions]):
        raise AssertionError(f"Research design contract broken: {metrics}")
    return metrics
