# Understanding the generated Pandas data files

The pipeline produces the following **files**, from which Pandas DataFrames can be reconstructed. Git itself cannot directly store a live Python DataFrame.

```python
import pandas as pd
weather = pd.read_parquet('data/processed/weather_clean.parquet')
features = pd.read_parquet('data/processed/feature_table.parquet')
forecast = pd.read_csv('outputs/test_predictions.csv', parse_dates=['forecast_origin', 'target_date'])
annual_weather = pd.read_csv('outputs/eda/annual_weather.csv')
annual_test_errors = pd.read_csv('outputs/error_analysis/annual_test_errors.csv')
heat_spells = pd.read_csv('outputs/error_analysis/observed_heat_episodes.csv')
extra_predictions = pd.read_csv('outputs/extra_models/test_predictions.csv')
```

- `outputs/eda/`: entirely descriptive history; evaluate coverage before inferring long-term climate trends.
- `outputs/error_analysis/`: final-test errors by year, and observed consecutive-day hot spells.
- `outputs/extra_models/`: Ridge / histogram boosting validation scores and held-out diagnostics; the winner is chosen from **validation**, not test.
- `outputs/research_report.md`: auto-generated transparent experiment writeup; headline identifies synthetic or DWD data.
- `outputs/figures/`, `outputs/eda/figures/`, `outputs/error_analysis/figures/`: PNG visualizations.

All generated data files and model artifacts are gitignored so repository history contains implementation, tests and documentation, not gigabytes of historical measurements. For full research reproducibility, record download versions and distribute a data-generation recipe.
