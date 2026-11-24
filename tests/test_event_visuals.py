import pandas as pd
from heat_forecasting.event_visuals import plot_heat_episode, extreme_day_diagnostics


def test_heat_event_visuals_and_extreme_case_table(tmp_path):
    dates=pd.date_range("2025-07-01",periods=5)
    p=pd.DataFrame({"target_date":dates,"target_tmax_c":[25,31,33,34,22],"random_forest":[26,29,30,31,26],"heat_alert":[0,1,1,0,1]})
    episodes=pd.DataFrame([{"start_date":"2025-07-02","end_date":"2025-07-04","peak_tmax_c":34.,"length_days":3}])
    assert plot_heat_episode(p,episodes,tmp_path/"heat_episode.png").exists()
    audit=extreme_day_diagnostics(p)
    assert int(audit.missed_alert.sum())==1
    assert int(audit.false_alarm.sum())==1
    assert plot_heat_episode(p,episodes.iloc[0:0],tmp_path/"empty.png") is None
