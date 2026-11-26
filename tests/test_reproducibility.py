import joblib
from heat_forecasting.reproducibility import sha256_file, save_fitted_models, run_provenance
from heat_forecasting.config import load_config
from heat_forecasting.demo import synthetic_daily_weather


def test_file_hash_and_model_persistence(tmp_path):
    p=tmp_path/"input.txt"
    p.write_text("hello")
    assert len(sha256_file(p))==64
    assert sha256_file(p)==sha256_file(p)
    paths=save_fitted_models({"toy": {"n":1}},tmp_path/"models")
    assert joblib.load(paths["toy"])=={"n":1}
    cfg=load_config("configs/default.yaml")
    _,metadata=synthetic_daily_weather(cfg)
    result=run_provenance(cfg,tmp_path,demo=True,station_metadata=metadata)
    assert result["input_sha256"]=={}
    assert "SYNTHETIC" in result["kind"]
