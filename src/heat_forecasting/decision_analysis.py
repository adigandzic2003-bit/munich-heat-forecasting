"""Decision-cost sensitivity, selecting each threshold from validation only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_columns
from .models import classifier_model, select_alert_threshold


def validation_cost_sensitivity(parts: dict[str, pd.DataFrame], cfg: dict, *, miss_costs: tuple[int, ...] = (1, 2, 4, 8)) -> pd.DataFrame:
    """Compare four illustrative alert costs, keeping test labels sealed."""
    train, valid = parts["train"], parts["validation"]
    if train.hot_event.nunique() < 2 or valid.hot_event.nunique() < 2:
        raise ValueError("Development datasets need both hot and non-hot days")
    features = feature_columns(train)
    classifier = classifier_model(cfg["model"]["classifier"], int(cfg["project"]["random_state"]))
    classifier.fit(train[features], train.hot_event.astype(int))
    probs = classifier.predict_proba(valid[features])[:, 1]
    labels = valid.hot_event.to_numpy(dtype=int)
    rows = []
    for cost in miss_costs:
        if cost <= 0:
            raise ValueError("Missed-event costs must be positive")
        candidate_cfg = {"model": dict(cfg["model"], false_alarm_cost=1.0, missed_event_cost=float(cost))}
        threshold, cost_grid = select_alert_threshold(labels, probs, candidate_cfg)
        closest = cost_grid.loc[np.isclose(cost_grid.threshold, threshold)].iloc[0]
        rows.append({"false_alarm_cost": 1.0, "missed_event_cost": float(cost),
                     "validation_selected_threshold": float(threshold),
                     "validation_false_alarms": int(closest.false_alarms),
                     "validation_missed_hot_days": int(closest.missed_events),
                     "validation_total_cost": float(closest.weighted_cost),
                     "validation_rows": len(valid)})
    return pd.DataFrame(rows)
