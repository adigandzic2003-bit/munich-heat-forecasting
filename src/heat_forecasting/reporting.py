"""Self-contained Markdown evidence report built only from executed experiment outputs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_research_report(
    path: str | Path, *, demo: bool, cfg: dict, test_predictions: pd.DataFrame,
    regression: pd.DataFrame, event: dict, annual_eda: pd.DataFrame,
    annual_errors: pd.DataFrame, heat_episodes: pd.DataFrame,
    additional_results: pd.DataFrame, additional_meta: dict,
) -> Path:
    """Write result-linked documentation. The synthetic flag is prominent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = regression.loc[regression.segment == "overall"].set_index("model")
    selected = str(additional_meta["selected_by_validation"])
    def score(name: str) -> str:
        return f"{float(overall.loc[name, 'mae_c']):.3f}" if name in overall.index else "n/a"

    sample_start = pd.Timestamp(test_predictions.target_date.min()).date()
    sample_end = pd.Timestamp(test_predictions.target_date.max()).date()
    hot_count = int(test_predictions.hot_event.sum())
    total_episodes = len(heat_episodes)
    detected = int(heat_episodes.any_alert.sum()) if total_episodes else 0
    source = "SYNTHETIC DEMONSTRATION – NOT EMPIRICAL EVIDENCE" if demo else "DWD HISTORICAL OBSERVATIONS – RETROSPECTIVE ANALYSIS"
    report = f"""# Munich heat-event forecasting: executed experiment report

> **{source}**

This report was generated from the actual executed Python pipeline, not manually typed figures. Results from the synthetic demonstration **must not be presented as scientific findings**.

## Problem definition and target

- Predict next-day maximum temperature at station `{cfg['data']['primary_station']}` (Munich).
- Hot day: observed daily maximum ≥ **{float(cfg['project']['hot_threshold_c']):g} °C**.
- Forecast origin: D at {int(cfg['project']['issue_hour_local']):02d}:00 Europe/Berlin, target D+1. Historical weather observations assumed available only through D-{int(cfg['project']['reporting_lag_days'])}.
- Test date range: **{sample_start} to {sample_end}**; {len(test_predictions)} test observations, including {hot_count} observed hot days.

## Models and held-out errors

| Method | Held-out test MAE (°C) |
|---|---:|
| Persistence | {score('persistence')} |
| Seasonal climatology | {score('climatology')} |
| Random forest (primary) | {score('random_forest')} |
| Without neighbors (ablation) | {score('ablation_no_neighbors')} |

Additional experiments (Ridge and histogram gradient boosting) use the **validation-only** selection rule. Selected supplemental model: **{selected}**. Their held-out scores are diagnostics, not a second test-based model-selection exercise.

```text\n{additional_results[['model', 'validation_mae_c', 'test_mae_c_diagnostic']].to_string(index=False)}\n```

## Hot-day decision quality

- True positives: **{event['true_positives']}**; false alarms: **{event['false_positives']}**.
- Missed hot days: **{event['false_negatives']}**; true negatives: **{event['true_negatives']}**.
- Precision: **{event['precision']:.3f}**; recall: **{event['recall']:.3f}**.
- Brier score: **{event['brier_score']:.4f}**. Inspect the calibration plot before treating probabilities as calibrated.
- Observed heat episodes of at least two consecutive days: **{total_episodes}**. At least one alert issued during **{detected}** of these episodes.

## Data quality and interpretation

- Years covered in descriptive data: **{int(annual_eda.year.min())}–{int(annual_eda.year.max())}**.
- Data coverage, yearly missing observations and temperature distribution are saved in `outputs/eda/`.
- Held-out error by year and episode-level detection are saved in `outputs/error_analysis/`.
- These ex-post descriptive tables include test-year outcomes; they are **not** used to select a model or change hyperparameters.

## Predefined study design

- Train through {cfg['split']['train_end']}; validation through {cfg['split']['validation_end']}; final independent test through {cfg['split']['test_end']}.
- Train-only imputation; chronological separation; no contemporaneous or future temperature used as predictors.
- Candidate regression settings and the alert threshold are selected on validation, then frozen on test.
- Controlled ablation removes neighboring-station features while leaving test dates and model settings fixed.

## Limitations and next human checks

1. DWD historical archives may be revised; record the archive hashes and retrieval date before submission.
2. Check DWD station moves, missingness, quality flags and the different microclimates of city versus airport stations.
3. Historical daily observations do not establish true operational publication times. The assumed D-1 availability at 18:00 must be independently checked.
4. Only three held-out calendar years are used; a block-bootstrap interval from three blocks is imprecise.
5. The project is retrospective, not a real-time weather warning service. Event alarms require further operational validation.
6. A team member must inspect the plots, investigate surprising results, and defend the experimental choices.

## Reproduce

`python -m heat_forecasting.pipeline --config configs/default.yaml {'--demo' if demo else ''}`

The generated CSV, Parquet, JSON and PNG files live in `data/processed/` and `outputs/`; the large derived artifacts are intentionally gitignored. 
"""
    path.write_text(report, encoding="utf-8")
    return path
