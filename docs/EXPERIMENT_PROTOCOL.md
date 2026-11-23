# Experiment protocol: Topic P4 – Munich next-day heat forecasting

## Scope and research claim

**Primary scientific question:** Can a random forest using lagged Munich and nearby station observations improve next-day maximum-temperature accuracy relative to persistence and seasonal climatology, and does it improve identification of 30°C days?

This study is **retrospective** and works with DWD *historical daily measurements*, not numerical weather prediction forecasts. The assumed origin is 18:00 local time, with the latest daily summaries from D-1. DWD publication latency remains unverified, so the outputs are not a real-time alert service.

## Pre-registered design

- Target station: `03379`, München-Stadt; neighboring features: `01262` and `05404`.
- Training target dates through 2019; validation through 2022; untouched final test: 2023–2025.
- Forecast D+1 maximum temperature; observed 30°C threshold defines a hot day.
- Predefined baselines: D-1 persistence and day-of-year climatology fitted on train-only data.
- Primary model: scikit-learn random forest; validation-only hyperparameter choice.
- Alert model: random-forest classifier with an alarm threshold chosen by validation-only weighted costs.
- Ablation: remove all neighbor station features while holding the model structure and evaluation dates constant.
- Metrics: MAE, RMSE, errors for actual hot days, event precision/recall, false alarms, Brier score and seasonal comparisons.
- Expanding-development backtests are reported separately from the final 2023–2025 held-out evaluation.

## Morning extensions (23 November 2026)

1. **Descriptive EDA:** Yearly weather coverage, missingness and observed heat-day counts; monthly temperature distribution. Descriptive EDA may report held-out periods but must never be used for model/threshold choices.
2. **Episode analysis:** Entire observed spells of at least two *consecutive calendar days* with temperature ≥30°C. Count the number of spells with at least one predicted alert and whole-year test errors. An absent calendar date breaks any spell.
3. **Supplementary algorithms:** Ridge regression and histogram gradient boosting, trained on the same causal feature set; select a *supplementary* candidate based solely on validation MAE, then refit and report its held-out diagnostic scores. **Do not replace or reselect the previously frozen primary RF based on held-out test performance.**
4. **Reproducible report:** Generate a Markdown experiment report and machine-readable CSV/JSON outputs directly from the code that ran. Label synthetic demo results explicitly.

## Avoiding leakage

- Time-index all station measurements to the complete daily calendar before shifts; never do row-position joins when dates are missing.
- All measured-weather inputs are taken from D-1 or earlier at the D origin date. Calendar features for the known target date are fine.
- Fit imputers and scalers **inside** sklearn Pipelines; never fit to final test features.
- Use train/validation only for model parameter and alert threshold selection.
- Use final test exclusively for scoring and ex-post diagnostics. Do not revisit test results repeatedly to choose winners.

## Data integrity and limitations

- Record source URL, station ID, archive date/version and cryptographic file checksums when real data are retrieved.
- Review quality flags and potential changes to DWD station locations before reporting causal/scientific interpretations.
- Missing observations alter denominators. The EDA records expected vs observed calendar days and valid temperatures.
- The limited three-year final test period makes year-block bootstrap uncertainty estimates unstable.
- Simple historical observations cannot represent operational 18:00 data availability without independent verification.
- The team's defense should focus on leakage, comparison baselines, scientific limitations and concrete failure cases.

## Execution

```bash
python -m pip install -e '.[test]'
pytest -q
python -m heat_forecasting.pipeline --config configs/default.yaml --demo
# After verifying network/data policy:
python -m heat_forecasting.pipeline --config configs/default.yaml
```

Review `outputs/research_report.md` alongside CSV tables and charts. Never quote synthetic scores as actual results.
