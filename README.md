# Munich Heat-Event Forecasting (Topic P4)

A reproducible, laptop-scale project for the TUM course *Programming in Python for Business and Life Science Analytics* (winter 2026/27). We use **historical** DWD daily weather observations to study prediction of tomorrow's maximum temperature and 30°C hot-day events in Munich.

> **Research prototype only.** Daily observation archives are NOT archived operational forecasts. The assumed observation reporting lag is explicit, but actual operational release latencies remain unverified. Do not use this as a public weather-warning service.

## Research question

How well do a random forest and historical-temperature baselines forecast next-day maximum temperature and hot days, and do neighboring weather stations improve accuracy compared with Munich-only lag features?

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
python -m heat_forecasting.pipeline --config configs/default.yaml --demo
python -m heat_forecasting.pipeline --config configs/default.yaml
```

- `--demo`: deterministic, **synthetic** 1993–2025 weather for offline proof of reproducibility; **not** valid research results.
- Without `--demo`: downloads real DWD historical archives into `data/raw/`; network access required. Repeated runs use the cached files.
- Derived `data/processed/*.parquet` tables load back into Pandas with `pd.read_parquet(...)`.
- `outputs/`: station metadata, QC tables, split manifest, test forecasts, baseline comparisons, backtest, event metrics, uncertainty, threshold optimization, and charts.
- `data/raw`, `data/processed`, `outputs` and model binaries are excluded from Git. Code and configuration are versioned, not large weather archives.

## Inputs and data provenance

| ID | Station | Role | Historical archive |
|---|---|---|---|
| 03379 | München-Stadt | Target | Available in DWD daily KL historical index |
| 01262 | München-Flughafen | Neighbor | Available in DWD daily KL historical index |
| 05404 | Weihenstephan-Dürnast | Neighbor | Available in DWD daily KL historical index |

Sources: [DWD historical archive index](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/), [DWD KL variable descriptions](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/DESCRIPTION_obsgermany-climate-daily-kl_en.pdf). The archive parser reads semicolon-delimited `produkt_klima_tag_*.txt`, preserves station/date IDs, converts missing `-999` to NaN, and retains metadata and original ZIP archives.

**Target:** daily maximum measured temperature `TXK` in degrees Celsius for München-Stadt on `origin_date + 1`; a hot day is `target_tmax_c >= 30°C`. These are meteorological demonstration labels.

## Forecast cutoff: explicitly prevent data leakage

Forecast origin D is hypothetically **18:00 local time**. The newest daily observations allowed in the feature table are from **D-1**, conservatively leaving one day of reporting latency. We predict **D+1**. We do *not* use temperature from day D or D+1 as a feature. If a real operational release at 18:00 requires more delay, increase `reporting_lag_days` in YAML and rerun the entire evaluation.

Features: Munich daily maximum at calendar lags 1, 2, 3, 7 and 14; trailing rolling averages from the available lagged history; lagged neighboring-station temperatures; cyclical day-of-year of the **target date** (known in advance). Reindexing to a complete daily calendar avoids silently shifting rows across missing dates. No imputation is fit on test data: median imputation is within the train-only sklearn pipeline.

## Splitting, experiments, and independent test

- **Train** on target dates through 2019-12-31.
- **Development/threshold validation** on 2020-01-01 through 2022-12-31.
- **Untouched final test** on 2023-01-01 through 2025-12-31.
- **Expanding backtests** over 2016–17, 2018–19, 2020–22 (pre-specified model settings; not independent estimates for selected hyperparameters).
- Model selection chooses between two small random-forest configurations using validation MAE, then refits on train+validation. No final test data drive hyperparameters or thresholds.
- **Required controlled ablation:** exact same dates and model settings, but remove all neighboring-station feature columns.
- Baselines: persistence from most recent allowed Munich observation (D-1), and smoothed day-of-year climatology fit only on training periods.
- Heat-event probabilities are predicted by an independent random-forest classifier. Choose the alert threshold on validation with explicit false-alarm cost 1 and missed-event cost 4, then freeze it on the final test. Probabilities are **evaluated for calibration**, not asserted to be calibrated.

## Evaluation and outputs

`outputs/test_predictions.csv` includes forecast origin and target, temperature labels and predictions, hot-event probabilities and alerts. `outputs/regression_scores.csv` has MAE and RMSE overall and by season; `outputs/heat_event_scores.json` has precision, recall, false alarms, average precision and Brier score; `outputs/calibration.csv` and `outputs/figures/calibration.png` provide probability calibration evidence. `outputs/development_backtest.csv` reports expanding-period baseline vs ML results. `outputs/uncertainty.json` contains a **calendar-year block bootstrap** interval for final-test MAE (only three blocks; interpret cautiously).

The station-identity and split manifests are reproducible and CSV/Parquet outputs can be loaded as DataFrames in any notebook.

## Project limitations and remaining validation

1. Source archives can have gaps, station moves, differing urban/airport microclimates, changing sensors and quality flags. Review `quality_audit.csv`, original DWD station metadata and any provenance warnings before scientific conclusions.
2. Target is a single urban site; neighboring stations are features, not additional prediction targets. The evaluation therefore reports per-season performance, not per-station test MAE. Extending to multiple target stations requires additional target-site experiments.
3. Archive ZIPs may change; cache them and report the date and hashes for a final submission. The actual source retrieval and full real-data run require internet and should be performed before accepting empirical metrics.
4. Three consecutive held-out years provide some temporal robustness, but uncertainty from three year-level blocks is imprecise. Report examples of entire heat episodes and consider a dedicated event-block analysis.
5. An actual 18:00 forecast needs verification that D-1 daily summaries were published then. Without this, these are **retrospective forecasts**, not deployable predictions.
6. The final report must explain all model choices, manually trace sample dates and add an informative failed experiment. Project team is responsible for understanding and validating every result, including AI-assisted code.

## Key project files

- `configs/default.yaml` – stations, target cutoff, split dates, hyperparameters, costs.
- `src/heat_forecasting/acquisition.py` – DWD ZIP download and parsing.
- `src/heat_forecasting/quality.py` – station/date uniqueness and missing-value auditing.
- `src/heat_forecasting/features.py` – calendar-aligned, leakage-safe Pandas features.
- `src/heat_forecasting/baselines.py` – persistence and seasonal climatology.
- `src/heat_forecasting/models.py` – temporal model selection, ablation, alert threshold, backtest.
- `src/heat_forecasting/evaluation.py` – event and temperature scores, calibration, bootstrap.
- `src/heat_forecasting/visualization.py` – charts.
- `src/heat_forecasting/pipeline.py` – end-to-end reproducible entrypoint.
- `tests/` – offline regression tests, including leakage and duplicate keys.

## Attribution and independent understanding

Substantial coding assistance: OpenAI ChatGPT (GPT-6), 22 November 2026. DWD data are independently sourced from the German Weather Service. ML uses scikit-learn. The team should review and be able to defend every line in a live demonstration.
