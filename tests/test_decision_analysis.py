from heat_forecasting.config import load_config
from heat_forecasting.demo import synthetic_daily_weather
from heat_forecasting.features import make_feature_table
from heat_forecasting.models import chronological_splits
from heat_forecasting.decision_analysis import validation_cost_sensitivity


def test_validation_only_cost_scenarios():
    cfg=load_config("configs/default.yaml")
    cfg["data"]["start_date"]="2010-01-01"
    cfg["model"]["classifier"]["n_estimators"]=8
    frame,_=synthetic_daily_weather(cfg)
    panel,_=make_feature_table(frame,cfg)
    parts=chronological_splits(panel,cfg)
    result=validation_cost_sensitivity(parts,cfg)
    assert set(result.missed_event_cost)=={1.0,2.0,4.0,8.0}
    assert result.validation_selected_threshold.between(0,1).all()
    assert result.validation_rows.eq(len(parts["validation"])).all()
