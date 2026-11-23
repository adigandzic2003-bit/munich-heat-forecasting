import numpy as np
import pandas as pd
from heat_forecasting.extra_models import additional_regressors, run_additional_regressors


def parts(seed=123):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2017-01-01', periods=350)
    frame = pd.DataFrame({
        'target_date': dates,
        'target_tmax_c': rng.normal(18, 7, size=len(dates)),
        'tmax_lag_1': rng.normal(17, 7, size=len(dates)),
        'day_sin': np.sin(2 * np.pi * dates.dayofyear / 365.25),
        'day_cos': np.cos(2 * np.pi * dates.dayofyear / 365.25),
    })
    return {'train': frame.iloc[:230].copy(), 'validation': frame.iloc[230:290].copy(), 'test': frame.iloc[290:].copy()}


def test_model_comparison_outputs_and_validation_selection():
    data = parts()
    original_test_target = data['test'].target_tmax_c.copy()
    preds, results, meta = run_additional_regressors(data, {'project': {'random_state': 42}})
    assert {'ridge', 'hist_gradient_boosting'}.issubset(preds.columns)
    assert not preds[['ridge', 'hist_gradient_boosting']].isna().any().any()
    assert len(preds) == len(data['test'])
    assert results.validation_mae_c.is_monotonic_increasing
    assert meta['selected_by_validation'] == results.iloc[0].model
    # Original test labels do not change, and the selection cannot look at them.
    assert data['test'].target_tmax_c.equals(original_test_target)


def test_cannot_select_on_test_labels():
    original = parts()
    _, results, meta = run_additional_regressors(original, {'project': {'random_state': 42}})
    modified = parts()
    modified['test']['target_tmax_c'] += 1000
    _, changed, new_meta = run_additional_regressors(modified, {'project': {'random_state': 42}})
    assert meta['selected_by_validation'] == new_meta['selected_by_validation']
    assert results.validation_mae_c.to_list() == changed.validation_mae_c.to_list()
