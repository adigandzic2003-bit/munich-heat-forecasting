import pandas as pd
import pytest
from heat_forecasting.config import load_config
from heat_forecasting.research_validation import audit_research_design


def test_research_contract_detects_disjoint_test_and_future_lag():
    cfg=load_config("configs/default.yaml")
    mk=lambda dates:pd.DataFrame({"target_date":pd.to_datetime(dates)})
    parts={"train":mk(["2019-01-02"]),"validation":mk(["2020-01-02"]),"test":mk(["2023-01-02"])}
    p=mk(["2023-01-02"])
    for c in ["random_forest","ablation_no_neighbors","heat_probability","heat_alert"]:p[c]=1
    feat=pd.DataFrame({"forecast_origin":pd.to_datetime(["2023-01-01"]),"target_date":pd.to_datetime(["2023-01-02"])})
    assert audit_research_design(parts,p,cfg,feature_table=feat)["prediction_test_dates_exact_match"]
    p["target_date"]=pd.to_datetime(["2025-01-02"])
    with pytest.raises(AssertionError):audit_research_design(parts,p,cfg,feature_table=feat)
