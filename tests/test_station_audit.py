import pandas as pd
from heat_forecasting.config import load_config
from heat_forecasting.demo import synthetic_daily_weather
from heat_forecasting.quality import audit_and_clean
from heat_forecasting.station_audit import station_quality_summary, missing_runs


def test_station_quality_summary_and_shared_calendar():
    cfg = load_config("configs/default.yaml")
    cfg["data"]["start_date"] = "2020-01-01"
    cfg["data"]["end_date"] = "2020-01-09"
    raw, _ = synthetic_daily_weather(cfg)
    raw["quality_flag_temp"] = 3
    clean, _ = audit_and_clean(raw, start_date=cfg["data"]["start_date"], end_date=cfg["data"]["end_date"])
    summary, flags, overlap = station_quality_summary(raw, clean, cfg)
    assert len(summary) == 3 and len(overlap) == 3
    assert set(summary.valid_tmax_days) == {9}
    assert len(flags) == 3
    assert all(overlap.common_valid_tmax_days == 9)


def test_missing_runs_detect_gaps_on_calendar_not_row_positions():
    cfg = load_config("configs/default.yaml")
    cfg["data"]["start_date"] = "2020-01-01"
    cfg["data"]["end_date"] = "2020-01-08"
    raw, _ = synthetic_daily_weather(cfg)
    raw = raw.loc[~((raw.station_id == "03379") & raw.date.isin(pd.to_datetime(["2020-01-03","2020-01-04"])))].copy()
    clean, _ = audit_and_clean(raw, start_date=cfg["data"]["start_date"], end_date=cfg["data"]["end_date"])
    runs = missing_runs(clean,cfg,min_days=2)
    assert len(runs) == 1 and runs.iloc[0].missing_days == 2
