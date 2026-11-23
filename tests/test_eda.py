import numpy as np
import pandas as pd
import pytest
from heat_forecasting.eda import descriptive_weather_tables, plot_descriptive_weather


def minimal_cfg():
    return {"data": {"primary_station": "03379", "start_date": "2024-01-01", "end_date": "2024-01-05"},
            "project": {"hot_threshold_c": 30}}


def test_eda_calendar_coverage_and_missing_days():
    observed = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
                             "station_id": ["03379"] * 3, "tmax_c": [29.0, 31.0, np.nan]})
    annual, monthly = descriptive_weather_tables(observed, minimal_cfg())
    row = annual.iloc[0]
    assert row.calendar_days == 5
    assert row.recorded_days == 3
    assert row.valid_temperature_days == 2
    assert row.observed_hot_days == 1
    assert row.coverage_fraction == 0.4
    assert row.missing_calendar_days == 2
    assert monthly.iloc[0].mean_tmax_c == 30


def test_eda_rejects_duplicate_date():
    observed = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 2), "station_id": ["03379"] * 2, "tmax_c": [1, 2]})
    with pytest.raises(ValueError, match="exactly one row"):
        descriptive_weather_tables(observed, minimal_cfg())


def test_eda_plot_exports(tmp_path):
    observed = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "station_id": ["03379"] * 5, "tmax_c": [1, 5, 9, 13, 31]})
    annual, monthly = descriptive_weather_tables(observed, minimal_cfg())
    assert all((tmp_path / p.split("/")[-1]).exists() for p in plot_descriptive_weather(annual, monthly, tmp_path))
