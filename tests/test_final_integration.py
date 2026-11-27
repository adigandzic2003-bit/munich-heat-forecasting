import importlib.util
import pandas as pd

from heat_forecasting.config import load_config
from heat_forecasting.pipeline import run


def test_complete_research_outputs(tmp_path, monkeypatch):
    if importlib.util.find_spec('pyarrow') is None:
        monkeypatch.setattr(pd.DataFrame,'to_parquet',lambda self,path,index=False:self.to_csv(str(path)+'.csv',index=index))
    cfg=load_config('configs/default.yaml')
    cfg['data']['start_date']='2010-01-01'
    cfg['model']['regressor_candidates']=[{'n_estimators':8,'min_samples_leaf':2,'max_features':.8}]
    cfg['model']['classifier']['n_estimators']=8
    cfg['model']['bootstrap_repetitions']=10
    cfg['split']['backtest']=[{'train_end':'2019-12-31','validation_end':'2021-12-31'}]
    run(cfg,demo=True,root=tmp_path)
    root=tmp_path/'outputs'
    required=['station_audit/station_quality.csv','station_audit/raw_quality_codes.csv',
              'station_audit/pairwise_overlap.csv','station_audit/missing_runs.csv',
              'station_comparison/seasonal_station_scores.csv','station_comparison/aligned_predictions.csv',
              'decision_analysis/validation_cost_sensitivity.csv',
              'error_analysis/extreme_day_diagnostics.csv','example_forecast.json',
              'research_checks.json','provenance.json','research_report.md',
              'model_artifacts.json','figures/forecast_cutoff_example.png']
    for relative in required:
        assert (root/relative).exists(),relative
    text=(root/'research_report.md').read_text()
    assert 'SYNTHETIC TEST DATA' in text
    assert 'station and season' in text
    import json
    artifacts=json.loads((root/'model_artifacts.json').read_text())
    assert len(artifacts)>=3
    assert all((tmp_path/path).exists() for path in artifacts.values())
