import pandas as pd
from heat_forecasting.final_report import append_evidence_appendix


def test_evidence_report_explicitly_labels_synthetic_results(tmp_path):
    report=tmp_path/"report.md"
    report.write_text("# Generated baseline report\n")
    example={"forecast_issued_date":"2023-01-01","forecast_issued_hour_local":18,
      "latest_allowed_observation_date":"2022-12-31","forecast_target_date":"2023-01-02",
      "predicted_tmax_c":7.,"observed_tmax_c_ex_post":8.,"heat_probability":.01,"heat_alert":False}
    q=pd.DataFrame([{"station_id":"03379","name":"Munich","valid_tmax_days":10,"expected_days":10,"valid_fraction":1.0}])
    s=pd.DataFrame([{"station_id":"03379","season":"summer","method":"random_forest","n_days":10,"mae_c":2.,"rmse_c":3.,"hot_day_mae_c":2.}])
    x=pd.DataFrame([{"target_date":"2023-01-02","target_tmax_c":8.,"random_forest":7.,"missed_alert":False,"false_alarm":False}])
    append_evidence_appendix(report,demo=True,station_quality=q,station_scores=s,threshold_costs=pd.DataFrame(),
                             example=example,design_checks={"chronological_disjoint_order":True},
                             provenance={"effective_config_sha256":"abc"},extreme_days=x,station_gaps=pd.DataFrame())
    r=report.read_text()
    assert "SYNTHETIC TEST DATA" in r
    assert "Latest allowed daily observation" in r
    assert "station descriptions" in r
