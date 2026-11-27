# Architecture and responsibilities

Course: Programming in Python for Business and Life Science Analytics (TUM), Topic P4.

## Study question

Does adding lagged measurements from neighboring DWD stations improve prediction of Munich's next-day daily maximum temperature and unusually hot days, when all variables respect an explicit forecast issue cutoff?

## Data flow

```
DWD historic daily KL archives (three station ZIPs)
    -> acquisition.py: preserve ZIPs, parse IDs/dates/TXK/QN fields
    -> quality.py + station_audit.py: unique station/date, missingness and flags
    -> features.py: daily calendar alignment; D-1 or older weather predictors
    -> chronological_splits: fixed 1993–2019 / 2020–22 / 2023–25
    -> baselines.py: persistence + train-only day-of-year climatology
    -> models.py: Random Forest regression, hot-day classifier, spatial ablation
    -> evaluation.py: MAE/RMSE, event recall, false alarms, calibration
    -> supplementary analyses:
       eda.py, error_analysis.py, extra_models.py,
       station_experiments.py, decision_analysis.py, forecast_demo.py,
       station_audit.py, event_visuals.py
    -> pipeline.py: CSV, Parquet, JSON, figures, joblib, research report
```

## Data model

- Raw observations: `date, station_id, tmax_c`, plus available official DWD fields and quality codes.
- Unique key: `(station_id, date)`; never merge measurements across different calendar years merely by month/day.
- Feature row: `forecast_origin=D`, `target_date=D+1`; only past temperature fields with lag `>= reporting_lag_days`.
- Held-out prediction row: actual max, predicted max by baseline and RF/ablation, observed hot-day indicator, probability and frozen alert.

`data/raw/` is an immutable cached ZIP location. `data/processed/*.parquet` is derived. Neither goes to Git. Tables are restored with `pd.read_parquet`.

## Controlled comparisons

1. Persistence, climatology and RF on identical held-out Munich target dates.
2. Same RF settings and target dates with and without neighbor features (required ablation).
3. Supplemental Ridge and Histogram Gradient Boosting chosen only on development validation.
4. Fixed-parameter RF on all three stations' **intersection** of held-out dates; report station × season MAE/RMSE.
5. Cost-sensitivity curves with thresholds chosen only on validation; principal alert rule held fixed.

## Research constraints

- Synthetic demos validate code paths only, not forecast skill.
- Observed daily DWD archives are not archived operational forecasts; issue-time release latency remains an unverified assumption.
- The held-out 2023–25 years are reserved for final reporting, never model selection.
- DWD source revisions, gaps, QN flags and station moves need human review.
