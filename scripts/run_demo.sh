#!/usr/bin/env bash
set -euo pipefail
python -m heat_forecasting.pipeline --config configs/default.yaml --demo
