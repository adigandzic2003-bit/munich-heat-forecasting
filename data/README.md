# Data policy

`data/raw/` contains downloaded DWD ZIP archives (not committed). Original ZIP bytes are left unchanged.
`data/processed/` contains generated Pandas/Parquet tables (not committed).

To reproduce, run `python -m heat_forecasting.pipeline --config configs/default.yaml` with internet access.

For a quick smoke test without internet, use `python -m heat_forecasting.pipeline --demo`. All DEMO observations are synthetic and **must never** be used as evidence in the real-life results or report.

Data origin: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/
Format: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/DESCRIPTION_obsgermany-climate-daily-kl_en.pdf
