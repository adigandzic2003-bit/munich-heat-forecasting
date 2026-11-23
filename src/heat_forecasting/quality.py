"""Station-level quality audit; no future-informed interpolation."""
from __future__ import annotations

import pandas as pd
import numpy as np


def audit_and_clean(frame: pd.DataFrame, *, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = {"date", "station_id", "tmax_c"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Missing columns {needed - set(frame.columns)}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["station_id"] = work["station_id"].astype(str).str.zfill(5)
    work = work.loc[work["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    audits = []
    cleaned = []
    for sid, group in work.groupby("station_id", sort=True):
        group = group.sort_values("date").copy()
        duplicate_count = int(group.duplicated(["date", "station_id"]).sum())
        # Identical duplicate rows are safe to collapse; conflicting duplicates are fatal.
        if duplicate_count:
            repeated = group[group.duplicated(["date", "station_id"], keep=False)]
            for _, records in repeated.groupby("date"):
                if len(records.drop_duplicates()) > 1:
                    raise ValueError(f"Conflicting duplicate records for {sid} on {records.iloc[0]['date']}")
            group = group.drop_duplicates(["date", "station_id"])
        outlier = group["tmax_c"].notna() & ~group["tmax_c"].between(-50, 55)
        group.loc[outlier, "tmax_c"] = np.nan
        audits.append({
            "station_id": sid, "first_date": str(group.date.min().date()),
            "last_date": str(group.date.max().date()), "rows": len(group),
            "duplicate_rows_collapsed": duplicate_count,
            "outlier_tmax_masked": int(outlier.sum()),
            "missing_tmax": int(group.tmax_c.isna().sum()),
            "missing_fraction_tmax": round(float(group.tmax_c.isna().mean()), 5),
            "missing_calendar_days": len(pd.date_range(group.date.min(), group.date.max())) - len(group),
        })
        cleaned.append(group)
    if not cleaned:
        raise ValueError("No observations remain within requested date range")
    combined = pd.concat(cleaned, ignore_index=True)
    if combined.duplicated(["date", "station_id"]).any():
        raise AssertionError("Date/station keys must be unique after quality audit")
    return combined, pd.DataFrame(audits)
