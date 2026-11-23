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
    save_json(output_dir / "run_manifest.json", {"data_kind": "synthetic" if demo else "real DWD archived observations", "figures": figures, "n_observations": len(clean), "n_feature_rows": len(panel), "split_manifest": manifest})
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
