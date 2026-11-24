import pandas as pd
from heat_forecasting.config import load_config
from heat_forecasting.demo import synthetic_daily_weather
from heat_forecasting.station_experiments import compare_target_stations


def test_all_stations_use_same_heldout_dates():
    cfg=load_config("configs/default.yaml")
    cfg["data"]["start_date"]="2018-01-01"
    cfg["data"]["end_date"]="2024-03-01"
    cfg["model"]["regressor_candidates"]=[{"n_estimators": 8,"min_samples_leaf": 3,"max_features":.8}]
    raw,_=synthetic_daily_weather(cfg)
    predictions,scores=compare_target_stations(raw,cfg)
    assert predictions.station_id.nunique()==3
    assert predictions.groupby("station_id").size().nunique()==1
    assert set(scores.method)=={"persistence","climatology","random_forest"}
    assert predictions.groupby("station_id").target_date.nunique().nunique()==1
    assert (scores.n_days>0).all()
