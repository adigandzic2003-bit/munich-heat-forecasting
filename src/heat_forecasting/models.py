"""Leakage-safe model selection, event threshold selection, and final refitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .baselines import SeasonalClimatology, persistence_predict
from .features import feature_columns


def chronological_splits(frame: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    times = pd.to_datetime(frame["target_date"])
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    valid_end = pd.Timestamp(cfg["split"]["validation_end"])
    test_end = pd.Timestamp(cfg["split"]["test_end"])
    if not train_end < valid_end < test_end:
        raise ValueError("Must have train_end < validation_end < test_end")
    result = {
        "train": frame.loc[times <= train_end].copy(),
        "validation": frame.loc[(times > train_end) & (times <= valid_end)].copy(),
        "test": frame.loc[(times > valid_end) & (times <= test_end)].copy(),
    }
    if any(part.empty for part in result.values()):
        raise ValueError("Chronological split produced an empty part; check date coverage")
    return result


def regression_model(params: dict, random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("forest", RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)),
    ])


def classifier_model(params: dict, random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("forest", RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced_subsample", **params)),
    ])


def select_regression_params(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: dict) -> tuple[dict, pd.DataFrame]:
    seed = int(cfg["project"]["random_state"])
    rows = []
    for params in cfg["model"]["regressor_candidates"]:
        fitted = regression_model(params, seed).fit(train[features], train["target_tmax_c"])
        pred = fitted.predict(valid[features])
        rows.append({**params, "validation_mae": float(np.mean(np.abs(valid["target_tmax_c"] - pred)))})
    result = pd.DataFrame(rows).sort_values("validation_mae").reset_index(drop=True)
    winning = {key: result.loc[0, key] for key in cfg["model"]["regressor_candidates"][0]}
    for key in ("n_estimators", "min_samples_leaf"):
        winning[key] = int(winning[key])
    return winning, result


def select_alert_threshold(labels: np.ndarray, probabilities: np.ndarray, cfg: dict) -> tuple[float, pd.DataFrame]:
    """Select threshold once on validation, using explicitly declared teaching costs."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    false_alarm = float(cfg["model"]["false_alarm_cost"])
    miss = float(cfg["model"]["missed_event_cost"])
    step = float(cfg["model"]["threshold_grid_step"])
    rows = []
    for threshold in np.arange(step, 1.0, step):
        alerts = probabilities >= threshold
        fp = int(((alerts == 1) & (labels == 0)).sum())
        fn = int(((alerts == 0) & (labels == 1)).sum())
        rows.append({"threshold": float(np.round(threshold, 5)), "false_alarms": fp, "missed_events": fn, "weighted_cost": false_alarm * fp + miss * fn})
    if not rows:
        raise ValueError("threshold_grid_step must be between 0 and 1")
    scores = pd.DataFrame(rows).sort_values(["weighted_cost", "threshold"], ascending=[True, False]).reset_index(drop=True)
    return float(scores.loc[0, "threshold"]), scores


def train_final_models(parts: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    train, valid, test = (parts[name] for name in ("train", "validation", "test"))
    seed = int(cfg["project"]["random_state"])
    features_full = feature_columns(train)
    features_ablated = feature_columns(train, include_neighbors=False)
    selected, selection_table = select_regression_params(train, valid, features_full, cfg)
    combined = pd.concat([train, valid], ignore_index=True)
    prediction = test[["forecast_origin", "target_date", "target_tmax_c", "hot_event"]].copy()
    lag = int(cfg["project"]["reporting_lag_days"])
    prediction["persistence"] = persistence_predict(test, reporting_lag_days=lag)
    prediction["climatology"] = SeasonalClimatology().fit(combined).predict(test)
    fit_models = {}
    for name, features in (("random_forest", features_full), ("ablation_no_neighbors", features_ablated)):
        regressor = regression_model(selected, seed)
        regressor.fit(combined[features], combined["target_tmax_c"])
        prediction[name] = regressor.predict(test[features])
        fit_models[name] = regressor
    clf_params = cfg["model"]["classifier"]
    if train["hot_event"].nunique() < 2:
        raise ValueError("Training dataset has only one event class; use a longer climate period")
    classifier_dev = classifier_model(clf_params, seed).fit(train[features_full], train["hot_event"].astype(int))
    val_prob = classifier_dev.predict_proba(valid[features_full])[:, 1]
    threshold, cost_table = select_alert_threshold(valid["hot_event"], val_prob, cfg)
    classifier_final = classifier_model(clf_params, seed).fit(combined[features_full], combined["hot_event"].astype(int))
    prediction["heat_probability"] = classifier_final.predict_proba(test[features_full])[:, 1]
    prediction["heat_alert"] = (prediction["heat_probability"] >= threshold).astype(int)
    prediction["heat_alert_rf_temperature"] = (prediction["random_forest"] >= cfg["project"]["hot_threshold_c"]).astype(int)
    fit_models["heat_classifier"] = classifier_final
    metadata = {
        "selected_regressor": selected, "classifier_parameters": clf_params,
        "chosen_alert_threshold": threshold,
        "threshold_selection": "minimum weighted cost on validation (2020–2022), frozen for test",
        "false_alarm_cost": cfg["model"]["false_alarm_cost"],
        "missed_event_cost": cfg["model"]["missed_event_cost"],
        "feature_columns": features_full, "ablation_columns": features_ablated,
        "final_model_fit_period": f"up to {cfg['split']['validation_end']}",
    }
    return prediction, selection_table, metadata, {"models": fit_models, "cost_table": cost_table}


def expanding_backtest(frame: pd.DataFrame, cfg: dict, regressor_params: dict) -> pd.DataFrame:
    """Date-aligned expanding-window development folds, never touching held-out years."""
    seed = int(cfg["project"]["random_state"])
    features = feature_columns(frame)
    rows = []
    dates = pd.to_datetime(frame["target_date"])
    for fold_idx, fold in enumerate(cfg["split"]["backtest"], start=1):
        train = frame.loc[dates <= pd.Timestamp(fold["train_end"])]
        valid = frame.loc[(dates > pd.Timestamp(fold["train_end"])) & (dates <= pd.Timestamp(fold["validation_end"]))]
        if train.empty or valid.empty:
            continue
        clf = regression_model(regressor_params, seed).fit(train[features], train["target_tmax_c"])
        methods = {
            "persistence": persistence_predict(valid, reporting_lag_days=int(cfg["project"]["reporting_lag_days"])),
            "climatology": SeasonalClimatology().fit(train).predict(valid),
            "random_forest": clf.predict(valid[features]),
        }
        for name, values in methods.items():
            error = valid["target_tmax_c"].to_numpy() - values
            rows.append({"fold": fold_idx, "train_through": fold["train_end"], "validation_through": fold["validation_end"],
                         "method": name, "n_dates": len(valid), "mae": float(np.mean(np.abs(error))),
                         "rmse": float(np.sqrt(np.mean(error ** 2)))})
    if not rows:
        raise ValueError("No usable backtest folds")
    return pd.DataFrame(rows)
