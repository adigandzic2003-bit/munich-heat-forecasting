import pandas as pd
import pytest
from heat_forecasting.forecast_demo import example_forecast, plot_forecast_example
from heat_forecasting.config import load_config


def test_cutoff_sample_and_plot(tmp_path):
    cfg = load_config("configs/default.yaml")
    dates = pd.date_range("2023-01-04", periods=5)
    frame = pd.DataFrame({"target_date": dates, "forecast_origin": dates-pd.Timedelta(days=1),
                          "random_forest": [9.,10.,11.,12.,13.], "persistence": [8.,9.,10.,11.,12.],
                          "target_tmax_c": [10.,10.,12.,11.,14.], "heat_probability": [.1]*5,
                          "heat_alert": [0]*5})
    sample = example_forecast(frame,cfg)
    assert sample["forecast_target_date"] == "2023-01-06"
    assert sample["latest_allowed_observation_date"] == "2023-01-04"
    assert plot_forecast_example(frame,sample,tmp_path/"example.png").exists()
    with pytest.raises(ValueError):
        example_forecast(frame,cfg,target_date="2023-01-11")
