import importlib.util
import pandas as pd

from heat_forecasting.config import load_config
from heat_forecasting.pipeline import run


def test_all_morning_outputs_generate_without_leakage(tmp_path, monkeypatch):
    # Local development environments may lack pyarrow; GitHub CI installs the
    # declared Parquet dependency and tests real parquet writing.
    if importlib.util.find_spec("pyarrow") is None:
        monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: self.to_csv(str(path) + ".csv", index=index))
    cfg = load_config("configs/default.yaml")
    cfg["data"]["start_date"] = "2010-01-01"
    cfg["model"]["regressor_candidates"] = [{"n_estimators": 12, "min_samples_leaf": 3, "max_features": .8}]
    cfg["model"]["classifier"]["n_estimators"] = 12
    cfg["model"]["bootstrap_repetitions"] = 15
    cfg["split"]["backtest"] = [{"train_end": "2019-12-31", "validation_end": "2021-12-31"}]
    result = run(cfg, demo=True, root=tmp_path)
    output = tmp_path / "outputs"
    expected = ["eda/annual_weather.csv", "eda/monthly_climatology.csv",
                "error_analysis/annual_test_errors.csv", "error_analysis/observed_heat_episodes.csv",
                "extra_models/test_predictions.csv", "extra_models/validation_and_test_scores.csv",
                "extra_models/selection_manifest.json", "research_report.md"]
    for filename in expected:
        assert (output / filename).exists(), filename
    assert "SYNTHETIC DEMONSTRATION" in (output / "research_report.md").read_text()
    extra = pd.read_csv(output / "extra_models/test_predictions.csv")
    assert extra.shape[0] == len(result["predictions"])
    assert extra.target_date.to_list() == pd.to_datetime(result["predictions"].target_date).dt.strftime("%Y-%m-%d").to_list()
