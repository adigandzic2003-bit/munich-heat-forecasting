import pandas as pd
import pytest

from heat_forecasting.quality import audit_and_clean


def test_quality_collapses_identical_duplicates_and_masks_outlier():
    df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-03']),
                       'station_id': ['03379'] * 3, 'tmax_c': [10., 10., 900.]})
    cleaned, audit = audit_and_clean(df, start_date='2020-01-01', end_date='2020-01-03')
    assert len(cleaned) == 2
    assert audit.loc[0, 'duplicate_rows_collapsed'] == 1
    assert audit.loc[0, 'missing_calendar_days'] == 1
    assert pd.isna(cleaned.loc[cleaned.date == pd.Timestamp('2020-01-03'), 'tmax_c']).all()


def test_conflicting_duplicates_fail_closed():
    df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01'] * 2),
                       'station_id': ['03379'] * 2, 'tmax_c': [10., 12.]})
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        audit_and_clean(df, start_date='2020-01-01', end_date='2020-01-02')
