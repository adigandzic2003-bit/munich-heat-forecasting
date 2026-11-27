# Results interpretation guide

The pipeline generates an evidence-linked report; this file is an interpretation template, **not a statement of measured results**.

## 1. Numerical forecast quality

Compare held-out MAE and RMSE for persistence, climatology, full RF and spatially ablated RF on identical test rows. Only claim a neighboring-station benefit if the difference is supported by the actual DWD test results and plausible across seasons. Note that three annual uncertainty blocks provide limited precision.

## 2. Hot days and decision costs

Look at recall, precision, false alarms, missed events, calibration and Brier score. Hottest-day MAE may worsen even when overall MAE improves. The 1:4 false-alarm/missed-event cost is a **teaching assumption**, not an official public-health risk standard.

## 3. Whole-event failures

Inspect all observed consecutive-day episodes, plus the selected illustrated episode. Discuss heat-onset and heat-decay errors, not only average daily errors. Selecting a figure ex-post is acceptable for illustration but not for tuning.

## 4. Temporal and station limitations

Climate conditions and reporting practices can change over time. Compare city, airport and Weihenstephan station conditions without assuming that geographic similarity makes the distributions identical. Retrospective daily summaries do not validate a live forecast product.

## 5. Discussion structure

Problem → datasets and provenance → information cutoff → baselines → split and backtesting → RF and ablation → heat-event metrics → uncertainty/limitation → cautious conclusion. All empirical numbers must be lifted from one completed real-data run.
