"""Add transparent scientific evidence tables to the executed research report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_evidence_appendix(path: str | Path, *, demo: bool, station_quality: pd.DataFrame,
                             station_scores: pd.DataFrame, threshold_costs: pd.DataFrame,
                             example: dict, design_checks: dict, provenance: dict,
                             extreme_days: pd.DataFrame, station_gaps: pd.DataFrame) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Primary research report must be generated first")
    q = station_quality[["station_id", "name", "valid_tmax_days", "expected_days", "valid_fraction"]]
    ss = station_scores.loc[(station_scores.method == "random_forest")].copy()
    ss = ss[["station_id", "season", "n_days", "mae_c", "rmse_c", "hot_day_mae_c"]]
    top = extreme_days.head(5)[["target_date", "target_tmax_c", "random_forest", "missed_alert", "false_alarm"]]
    def table(df: pd.DataFrame) -> str:
        return df.to_string(index=False, na_rep="n/a") if not df.empty else "No observations available."
    status = "SYNTHETIC TEST DATA – NO SCIENTIFIC CONCLUSIONS" if demo else "OBSERVED DWD DATA – RETROSPECTIVE PROTOTYPE"
    addition = f"""

## Reproducibility and additional evidence appendix

> **{status}**. Every table below is computed from the same executed pipeline run.

### Station identity and measurement coverage

The station information below comes from the project configuration; it must be
cross-checked against the DWD station descriptions before publication.

```text
{table(q)}
```

Unobserved-calendar gaps of at least two days: **{len(station_gaps)}**.
The raw quality-flag codes are tabulated separately: they have **not** been
interpreted as good/bad without the official field-specific DWD definitions.

### Test MAE/RMSE by target station and season

Fixed regression hyperparameters chosen before examining held-out target-station
data. All target sites use the same intersected held-out dates.

```text
{table(ss)}
```

### Validation-only hot-day alert decision scenarios

The purpose of these alternative scenarios is to demonstrate the cost tradeoff,
not to maximize a test-set metric. The principal alert threshold is set from
development data with the predeclared missed-event cost from the YAML file.

```text
{table(threshold_costs)}
```

### One auditable issued forecast

- Forecast origin: **{example['forecast_issued_date']}**, {example['forecast_issued_hour_local']:02d}:00 local time.
- Latest allowed daily observation: **{example['latest_allowed_observation_date']}**.
- Target date (D+1): **{example['forecast_target_date']}**.
- Predicted maximum: **{example['predicted_tmax_c']:.2f} °C**; observed ex-post: **{example['observed_tmax_c_ex_post']:.2f} °C**.
- Classifier's heat-day probability: **{example['heat_probability']:.3f}**; decision: **{example['heat_alert']}**.
- Historical archive release latency at the assumed issue time has **not** been independently confirmed.

### Extreme-day failure cases (ex-post diagnostic)

```text
{table(top)}
```

### Machine-checked study design

```text
{design_checks}
```

### Archive hashes and software environment

```text
{provenance}
```

### Human-review limitations

1. Inspect original DWD station descriptions for relocation/instrument changes;
   project-configured coordinates have not been verified automatically.
2. Check real DWD archives and field-specific quality-flag meanings against
   the official daily-KL documentation.
3. Verify that daily summaries were actually available at the assumed issue
   time. Until then, this is a retrospective experiment only.
4. Rare events may be concentrated in a few summers; three independent final
   years are not sufficient for definitive long-run risk claims.
5. Avoid using the held-out case study or extra test-score tables to adjust
   model architecture, feature selection, or operational alert thresholds.
"""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(addition)
    return path
