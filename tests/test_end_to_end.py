import pandas as pd

from heat_forecasting.config import load_config
from heat_forecasting.demo import synthetic_daily_weather
from heat_forecasting.features import make_feature_table
from heat_forecasting.quality import audit_and_clean
from heat_forecasting.models import chronological_splits, expanding_backtest, train_final_models
from heat_forecasting.evaluation import event_scores, regression_scores


def test_offline_synthetic_pipeline_core():
    cfg = load_config('configs/default.yaml')
    # Narrower offline sample for fast CI. The actual research run uses the YAML dates.
    cfg['data']['start_date'] = '2010-01-01'
    cfg['split']['backtest'] = [{'train_end': '2017-12-31', 'validation_end': '2018-12-31'}]
    cfg['model']['regressor_candidates'] = [{'n_estimators': 12, 'min_samples_leaf': 2, 'max_features': .8}]
    cfg['model']['classifier']['n_estimators'] = 12
    observations, _ = synthetic_daily_weather(cfg)
    clean, audit = audit_and_clean(observations, start_date=cfg['data']['start_date'], end_date=cfg['data']['end_date'])
    assert not audit.empty
    feature_frame, _ = make_feature_table(clean, cfg)
    parts = chronological_splits(feature_frame, cfg)
    assert parts['test']['target_date'].min().year == 2023
    prediction, _, meta, _ = train_final_models(parts, cfg)
    assert not prediction.isna().any().any()
    assert {'random_forest', 'ablation_no_neighbors', 'heat_probability'}.issubset(prediction.columns)
    assert 0 < meta['chosen_alert_threshold'] < 1
    assert event_scores(prediction, cfg)['n_days'] == len(prediction)
    assert not regression_scores(prediction).empty
    assert len(expanding_backtest(feature_frame.loc[feature_frame.target_date.dt.year <= 2022], cfg, meta['selected_regressor'])) == 3
