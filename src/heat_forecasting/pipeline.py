"""Single-entrypoint pipeline: acquisition → quality → features → models → evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .acquisition import fetch_historical
from .config import load_config, save_json
from .demo import synthetic_daily_weather
from .evaluation import calibration_table, event_scores, regression_scores, year_block_mae_interval
from .features import make_feature_table
from .models import chronological_splits, expanding_backtest, train_final_models
from .quality import audit_and_clean
from .visualization import make_figures
from .eda import descriptive_weather_tables, plot_descriptive_weather
from .error_analysis import heat_episodes, annual_test_errors, plot_annual_errors
from .extra_models import run_additional_regressors
from .reporting import write_research_report


def run(cfg: dict, *, demo: bool = False, root: str | Path = ".") -> dict:
    root = Path(root)
    data_dir = root / "data" / "processed"
    output_dir = root / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, metadata = synthetic_daily_weather(cfg) if demo else fetch_historical(cfg, root / "data" / "raw")
    clean, audit = audit_and_clean(raw, start_date=cfg["data"]["start_date"], end_date=cfg["data"]["end_date"])
    panel, rules = make_feature_table(clean, cfg)
    parts = chronological_splits(panel, cfg)
    preds, parameter_search, model_info, artifacts = train_final_models(parts, cfg)
    folds = expanding_backtest(panel.loc[pd.to_datetime(panel.target_date) <= pd.Timestamp(cfg["split"]["validation_end"])], cfg, cfg["model"]["regressor_candidates"][0])
    regression = regression_scores(preds, hot_threshold_c=float(cfg["project"]["hot_threshold_c"]))
    calibration = calibration_table(preds)
    scores = event_scores(preds, cfg)
    bootstrap = year_block_mae_interval(preds, cfg)
    figures = make_figures(preds, regression, calibration, output_dir / "figures", cfg)
    # Supplemental Morning Batch 2: descriptive studies and frozen-test diagnostics.
    # No result from these functions feeds back into the primary model selection.
    annual_weather, monthly_weather = descriptive_weather_tables(clean, cfg)
    eda_figures = plot_descriptive_weather(annual_weather, monthly_weather, output_dir / "eda" / "figures")
    annual_errors = annual_test_errors(preds, threshold_c=float(cfg["project"]["hot_threshold_c"]))
    episodes = heat_episodes(preds, threshold_c=float(cfg["project"]["hot_threshold_c"]), min_days=2)
    error_figures = plot_annual_errors(annual_errors, output_dir / "error_analysis" / "figures")
    extra_predictions, extra_scores, extra_metadata = run_additional_regressors(parts, cfg)
    (output_dir / "eda").mkdir(exist_ok=True)
    (output_dir / "error_analysis").mkdir(exist_ok=True)
    (output_dir / "extra_models").mkdir(exist_ok=True)
    annual_weather.to_csv(output_dir / "eda" / "annual_weather.csv", index=False)
    monthly_weather.to_csv(output_dir / "eda" / "monthly_climatology.csv", index=False)
    annual_errors.to_csv(output_dir / "error_analysis" / "annual_test_errors.csv", index=False)
    episodes.to_csv(output_dir / "error_analysis" / "observed_heat_episodes.csv", index=False)
    extra_predictions.to_csv(output_dir / "extra_models" / "test_predictions.csv", index=False)
    extra_scores.to_csv(output_dir / "extra_models" / "validation_and_test_scores.csv", index=False)
    save_json(output_dir / "extra_models" / "selection_manifest.json", extra_metadata)
    # Keep original downloaded archives separate; these are reproducible derived products.
    clean.to_parquet(data_dir / "weather_clean.parquet", index=False)
    panel.to_parquet(data_dir / "feature_table.parquet", index=False)
    metadata.to_csv(output_dir / "station_metadata.csv", index=False)
    audit.to_csv(output_dir / "quality_audit.csv", index=False)
    manifest = []
    for name, part in parts.items():
        manifest.append({"split": name, "first_target": str(part.target_date.min().date()), "last_target": str(part.target_date.max().date()), "n_rows": len(part)})
    pd.DataFrame(manifest).to_csv(output_dir / "split_manifest.csv", index=False)
    preds.to_csv(output_dir / "test_predictions.csv", index=False)
    parameter_search.to_csv(output_dir / "regressor_selection.csv", index=False)
    artifacts["cost_table"].to_csv(output_dir / "validation_threshold_costs.csv", index=False)
    folds.to_csv(output_dir / "development_backtest.csv", index=False)
    regression.to_csv(output_dir / "regression_scores.csv", index=False)
    calibration.to_csv(output_dir / "calibration.csv", index=False)
    save_json(output_dir / "feature_availability.json", rules)
    save_json(output_dir / "model_manifest.json", model_info)
    save_json(output_dir / "heat_event_scores.json", scores)
    save_json(output_dir / "uncertainty.json", bootstrap)
    write_research_report(output_dir / "research_report.md", demo=demo, cfg=cfg,
                          test_predictions=preds, regression=regression, event=scores,
                          annual_eda=annual_weather, annual_errors=annual_errors,
                          heat_episodes=episodes, additional_results=extra_scores,
                          additional_meta=extra_metadata)
    save_json(output_dir / "run_manifest.json", {"data_kind": "synthetic" if demo else "real DWD archived observations",
                                                     "figures": figures + eda_figures + error_figures,
                                                     "n_observations": len(clean), "n_feature_rows": len(panel),
                                                     "split_manifest": manifest,
                                                     "supplementary_validation_winner": extra_metadata["selected_by_validation"]})
    print("Pipeline complete:", "SYNTHETIC DEMO" if demo else "DWD DATA")
    print("Station/day rows:", len(clean), "Feature rows:", len(panel), "Test rows:", len(preds))
    print("Regression results:\n", regression.query("segment == 'overall'").to_string(index=False))
    print("Heat events:", scores)
    return {"predictions": preds, "regression": regression, "heat_event_scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description="Munich heat-event forecasting – reproducible research prototype")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--demo", action="store_true", help="Use explicitly synthetic weather instead of accessing the DWD server")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    run(load_config(args.config), demo=args.demo, root=args.root)


if __name__ == "__main__":
    main()
