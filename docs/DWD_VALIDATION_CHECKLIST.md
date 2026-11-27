# Real-data validation checklist (must complete before submission)

- [ ] Execute `python -m heat_forecasting.pipeline --config configs/default.yaml` against DWD daily-KL archives.
- [ ] Confirm all configured station IDs resolve in the current archive listing.
- [ ] Inspect `outputs/station_audit/station_quality.csv` for enough continuous coverage.
- [ ] Examine `outputs/station_audit/raw_quality_codes.csv`; check the codes' official DWD definitions. Do **not** assume every numeric code is a defective measurement.
- [ ] Compare configured coordinates/elevation and station identity against DWD `KL_Tageswerte_Beschreibung_Stationen.txt` and archive documentation; note any relocations or instrumentation changes.
- [ ] Check a handful of dates manually against the archived text, including sentinel `-999` and daily `TXK` values.
- [ ] Inspect `outputs/research_checks.json` and `outputs/feature_availability.json` before interpreting any MAE.
- [ ] Investigate strange chronology, summer-only missingness and gaps rather than replacing missing values using future outcomes.
- [ ] Open the held-out temperature timeline, heat-episode plot, calibration plot and station-by-season scores.
- [ ] Check if 18:00 CET forecasts really could use D-1 daily reports. Otherwise report the work as explicitly retrospective.
- [ ] Read `outputs/research_report.md`; use actual observed-data results, not synthetic demonstration figures, in the presentation.
- [ ] Record data retrieval date, ZIP SHA256 hashes and package versions in `outputs/provenance.json`.
- [ ] Have each team member explain the model selection, spatial ablation and validation/test split.

Source: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/
