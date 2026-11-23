import numpy as np
import pandas as pd
import pytest

from heat_forecasting.features import make_feature_table, feature_columns


def cfg():
    return {'project': {'reporting_lag_days': 1, 'horizon_days': 1, 'hot_threshold_c': 30},
            'data': {'start_date': '2020-01-01', 'end_date': '2020-01-16',
                     'primary_station': '03379', 'stations': {'03379': {}, '01262': {}},
                     'primary_lags': [1, 2, 3], 'neighbor_lags': [1], 'rolling_days': [3]}}


def test_lags_avoid_day_d_and_future_leakage():
    dates = pd.date_range('2020-01-01', periods=16)
    station = pd.DataFrame({'date': dates, 'station_id': '03379', 'tmax_c': np.arange(16, dtype=float)})
    neighbor = pd.DataFrame({'date': dates, 'station_id': '01262', 'tmax_c': np.arange(100, 116, dtype=float)})
    panel, rules = make_feature_table(pd.concat([station, neighbor]), cfg())
    row = panel.loc[panel.forecast_origin == pd.Timestamp('2020-01-10')].iloc[0]
    assert row['tmax_lag_1'] == 8  # January 9
    assert row['target_tmax_c'] == 10  # January 11
    assert row['neighbor_01262_lag_1'] == 108
    assert 'neighbor_01262_lag_1' not in feature_columns(panel, include_neighbors=False)
    assert 'target_tmax_c' not in feature_columns(panel)
    assert rules['latest_allowed_observation'] == 'D-1'


def test_gap_does_not_shift_neighbor_from_wrong_day():
    dates = pd.date_range('2020-01-01', periods=16)
    primary = pd.DataFrame({'date': dates, 'station_id': '03379', 'tmax_c': np.arange(16.)})
    neighbor = pd.DataFrame({'date': dates.delete(8), 'station_id': '01262', 'tmax_c': 3.})
    panel, _ = make_feature_table(pd.concat([primary, neighbor]), cfg())
    row = panel.loc[panel.forecast_origin == pd.Timestamp('2020-01-10')].iloc[0]
    assert pd.isna(row['neighbor_01262_lag_1'])
