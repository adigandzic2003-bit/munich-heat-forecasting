# P4 implementation checklist / Week of 23–27 November 2026

- [x] Deterministic synthetic test fixture and baseline pipeline.
- [x] DWD archive parser and Pandas/Parquet data model.
- [x] Chronological training/validation/test split and expanding backtests.
- [x] Persistence, climatology, Random Forest and required neighbor-feature ablation.
- [x] Temperature and hot-day metrics, Brier and calibration figures.
- [x] EDA, heat episodes and supplemental regression models.
- [x] Detailed station/date coverage and raw quality-code tables.
- [x] MAE/RMSE by target station and season on shared held-out dates.
- [x] Validation-only threshold cost-sensitivity table.
- [x] One illustrative issued forecast with explicit feature cutoff.
- [x] Versioned pipeline, model artifacts and SHA256 provenance manifest.
- [x] Machine-checkable research-design assertions and reproducible report.
- [ ] **Pending external verification:** Real DWD download and completed integration run.
- [ ] **Pending human review:** DWD quality-code meanings, station histories and assumed issue-time latency.
- [ ] **Pending human review:** Empirical interpretation, course presentation and oral defense.

GitHub Actions is best-effort; scheduled jobs can be delayed, even after the planned deadline. Use workflow_dispatch on the roadmap workflow if release milestones were missed.
