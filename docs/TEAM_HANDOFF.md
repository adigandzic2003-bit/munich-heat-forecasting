# Team handoff and demo

## Single command

```bash
python -m pip install -e '.[test]'
pytest -q
python -m heat_forecasting.pipeline --config configs/default.yaml --demo
python -m heat_forecasting.pipeline --config configs/default.yaml
```

The first run is a **synthetic offline smoke test**. The second downloads DWD data, creates `data/processed/*.parquet` and generates actual retrospective results under `outputs/`.

## What to demonstrate in front of the instructor

1. Show `configs/default.yaml` and explicitly state target, cutoff and splits.
2. Show the raw DWD ZIPs and `outputs/station_audit/` quality summaries.
3. Open `data/processed/feature_table.parquet` as a Pandas DataFrame and inspect date alignment.
4. Show persistence and climatology versus RF test metrics.
5. Explain the required spatial-feature ablation and the station-by-season comparison.
6. Open `outputs/example_forecast.json` and the corresponding plot; manually check the D-1 cutoff.
7. Show `outputs/decision_analysis/validation_cost_sensitivity.csv` and principal alert threshold.
8. Explain limitations, especially retrospective data availability.

## Division of review work

- Data owner: verify DWD station identity, source dates, units, quality flags and archive change history.
- Modeling owner: defend feature availability, refitting, validation-only model choices and classifier threshold.
- Evaluation owner: inspect event timelines, test-year errors, error by station/season and uncertainty.
- Report owner: replace any unverified wording, cite DWD, and keep synthetic figures out of scientific conclusions.

No one should submit a performance claim without executing and reviewing the DWD-data run.
