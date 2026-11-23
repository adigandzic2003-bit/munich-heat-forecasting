"""Train-only persistence and day-of-year climatology baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_predict(frame: pd.DataFrame, *, reporting_lag_days: int) -> np.ndarray:
    return frame[f"tmax_lag_{reporting_lag_days}"].to_numpy(dtype=float)


class SeasonalClimatology:
    """Training-data day-of-year means, smoothed in a ±14-day seasonal window."""

    def fit(self, train: pd.DataFrame) -> "SeasonalClimatology":
        # Leap day is mapped to day 365 for stable seasonal mapping.
        doy = np.minimum(pd.to_datetime(train["target_date"]).dt.dayofyear.to_numpy(), 365)
        y = train["target_tmax_c"].to_numpy(dtype=float)
        self.global_mean_ = float(np.nanmean(y))
        self.by_day_ = {}
        for day in range(1, 366):
            circular = np.minimum(np.abs(doy - day), 365 - np.abs(doy - day))
            values = y[circular <= 14]
            self.by_day_[day] = float(np.nanmean(values)) if values.size else self.global_mean_
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        doy = np.minimum(pd.to_datetime(frame["target_date"]).dt.dayofyear.to_numpy(), 365)
        return np.asarray([self.by_day_.get(int(day), self.global_mean_) for day in doy], dtype=float)
