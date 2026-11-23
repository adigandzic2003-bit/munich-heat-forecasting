import pandas as pd
import pytest
from heat_forecasting.error_analysis import heat_episodes, annual_test_errors, plot_annual_errors


def sample():
    return pd.DataFrame({
        "target_date": pd.to_datetime(["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-05", "2025-07-06", "2025-07-07"]),
        "target_tmax_c": [30, 32, 25, 31, 32, 29],
        "random_forest": [29, 31, 26, 30, 31, 30],
        "persistence": [28, 30, 30, 29, 30, 27],
        "heat_event": [1, 1, 0, 1, 1, 0],
        "heat_alert": [1, 0, 0, 0, 1, 1],
        "heat_probability": [.8, .4, .1, .3, .7, .6],
    })


def test_episodes_count_true_calendar_days_and_alerts():
    result = heat_episodes(sample(), min_days=2)
    assert len(result) == 2
    assert result.length_days.to_list() == [2, 2]
    assert result.any_alert.to_list() == [True, True]
    assert result.alerted_days.to_list() == [1, 1]
    assert result.missed_days.to_list() == [1, 1]
    assert heat_episodes(sample(), min_days=3).empty


def test_episode_missing_day_breaks_contiguity():
    frame = sample().loc[lambda x: x.target_date != pd.Timestamp('2025-07-02')]
    assert len(heat_episodes(frame, min_days=2)) == 1


def test_annual_errors_and_figure(tmp_path):
    annual = annual_test_errors(sample())
    assert annual.iloc[0].observed_hot_days == 4
    assert annual.iloc[0].false_alerts == 1
    assert annual.iloc[0].missed_hot_days == 2
    assert len(plot_annual_errors(annual, tmp_path)) == 1


def test_episodes_reject_duplicate_target_date():
    frame = pd.concat([sample(), sample().iloc[[0]]])
    with pytest.raises(ValueError, match='unique'):
        heat_episodes(frame)
