"""Supplementary reproducible regressors with validation-only model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import feature_columns


def additional_regressors(seed: int) -> dict[str, Pipeline]:
    """Two pre-registered supplementary candidates, deliberately small and CPU-friendly."""
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                random_state=seed, max_iter=120, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.0, early_stopping=False,
            )),
        ]),
    }


def run_additional_regressors(parts: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare candidates on validation; refit both on train+valid and evaluate test.

    The frozen *primary* RF model is never re-selected here. The test set is
    diagnostic only; the reported supplementary winner uses validation MAE.
    """
    train, valid, test = (parts[name] for name in ("train", "validation", "test"))
    features = feature_columns(train)
    if not features:
        raise ValueError("No model features found")
    seed = int(cfg["project"]["random_state"])
    train_valid = pd.concat([train, valid], ignore_index=True)
    rowlist: list[dict] = []
    output = test[["target_date", "target_tmax_c"]].copy().reset_index(drop=True)
    for name, model in additional_regressors(seed).items():
        model.fit(train[features], train.target_tmax_c)
        valid_pred = model.predict(valid[features])
        valid_mae = float(mean_absolute_error(valid.target_tmax_c, valid_pred))
        model.fit(train_valid[features], train_valid.target_tmax_c)
        test_pred = model.predict(test[features])
        output[name] = test_pred
        rowlist.append({
            "model": name,
            "validation_mae_c": valid_mae,
            "test_mae_c_diagnostic": float(mean_absolute_error(test.target_tmax_c, test_pred)),
            "test_rmse_c_diagnostic": float(np.sqrt(mean_squared_error(test.target_tmax_c, test_pred))),
            "n_features": len(features),
        })
    results = pd.DataFrame(rowlist).sort_values("validation_mae_c").reset_index(drop=True)
    meta = {"selected_by_validation": str(results.iloc[0].model),
            "selection_metric": "lowest 2020–2022 validation MAE; test scores diagnostic only",
            "features": features,
            "candidates": list(additional_regressors(seed)),
            "note": "Supplementary experiments: primary RF and heat-event threshold remain frozen"}
    return output, results, meta
