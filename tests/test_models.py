import numpy as np
import pandas as pd

from heat_forecasting.models import chronological_splits, select_alert_threshold
from heat_forecasting.baselines import SeasonalClimatology


def test_date_based_split_uses_target_date():
    frame = pd.DataFrame({'target_date': pd.to_datetime(['2019-12-31', '2020-01-01', '2023-01-01']),
                          'forecast_origin': pd.to_datetime(['2019-12-30', '2019-12-31', '2022-12-31'])})
    cfg = {'split': {'train_end': '2019-12-31', 'validation_end': '2022-12-31', 'test_end': '2025-12-31'}}
    parts = chronological_splits(frame, cfg)
    assert [len(parts[n]) for n in ['train', 'validation', 'test']] == [1, 1, 1]


def test_threshold_selection_penalizes_misses():
    cfg = {'model': {'false_alarm_cost': 1., 'missed_event_cost': 4., 'threshold_grid_step': .1}}
    threshold, table = select_alert_threshold(np.array([0, 0, 1, 1]), np.array([0.05, 0.2, 0.3, 0.85]), cfg)
    assert 0 < threshold <= .3
    assert len(table) > 0


def test_seasonal_climatology_only_fits_training_observations():
    train = pd.DataFrame({'target_date': pd.to_datetime(['2018-07-01', '2019-07-01']), 'target_tmax_c': [20., 30.]})
    value = SeasonalClimatology().fit(train).predict(pd.DataFrame({'target_date': pd.to_datetime(['2023-07-01'])}))[0]
    assert np.isclose(value, 25.)
