"""Transparent DWD station audits. Quality codes are *reported*, never guessed."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def station_quality_summary(raw: pd.DataFrame, clean: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return station summary, quality-code counts, pairwise daily overlap.

    DWD quality flags have dataset-specific definitions; we preserve them as-is
    and leave interpretation to the accompanying official DWD documentation.
    """
    wanted = {"date", "station_id", "tmax_c"}
    if wanted - set(raw) or wanted - set(clean):
        raise ValueError("Both inputs require date, station_id and tmax_c")
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])
    calendar = pd.date_range(start, end, freq="D")
    stations = cfg["data"]["stations"]
    summaries, flags = [], []
    valid_dates = {}
    for sid, definition in stations.items():
        raw_station = raw.loc[raw.station_id.astype(str).str.zfill(5).eq(sid)].copy()
        raw_station["date"] = pd.to_datetime(raw_station.date)
        raw_station = raw_station.loc[raw_station.date.between(start, end)]
        subset = clean.loc[clean.station_id.astype(str).str.zfill(5).eq(sid)].copy()
        subset["date"] = pd.to_datetime(subset.date)
        subset = subset.loc[subset.date.between(start, end)]
        if subset.duplicated("date").any():
            raise ValueError(f"Duplicated dates for station {sid}")
        recorded = subset.loc[subset.tmax_c.notna(), "date"]
        valid_dates[sid] = set(recorded)
        summaries.append({
            "station_id": sid, "name": definition["name"], "role": definition["role"],
            "latitude": definition["latitude"], "longitude": definition["longitude"],
            "elevation_m": definition["elevation_m"],
            "first_recorded_date": subset.date.min(), "last_recorded_date": subset.date.max(),
            "expected_days": len(calendar), "reported_rows": len(subset),
            "valid_tmax_days": int(subset.tmax_c.notna().sum()),
            "missing_calendar_days": len(calendar) - len(subset),
            "missing_or_invalid_tmax_days": len(calendar) - int(subset.tmax_c.notna().sum()),
            "valid_fraction": float(subset.tmax_c.notna().sum() / len(calendar)),
            "min_tmax_c": float(subset.tmax_c.min()) if subset.tmax_c.notna().any() else np.nan,
            "max_tmax_c": float(subset.tmax_c.max()) if subset.tmax_c.notna().any() else np.nan,
            "metadata_note": "Coordinates/elevation from the project configuration; independently verify against DWD station history",
        })
        for column in ("quality_flag_temp", "quality_flag_rain"):
            if column not in raw_station:
                continue
            counts = raw_station[column].fillna("MISSING").astype(str).value_counts(dropna=False)
            flags.extend({"station_id": sid, "field": column, "raw_code": code, "n_records": int(count)}
                         for code, count in counts.items())
    pairs = [{"station_a": a, "station_b": b, "common_valid_tmax_days": len(valid_dates[a] & valid_dates[b]),
              "fraction_of_full_calendar": len(valid_dates[a] & valid_dates[b]) / len(calendar)}
             for a, b in combinations(stations, 2)]
    return pd.DataFrame(summaries), pd.DataFrame(flags, columns=["station_id", "field", "raw_code", "n_records"]), pd.DataFrame(pairs, columns=["station_a", "station_b", "common_valid_tmax_days", "fraction_of_full_calendar"])


def missing_runs(clean: pd.DataFrame, cfg: dict, *, min_days: int = 2) -> pd.DataFrame:
    """Identify contiguous missing observations in a calendar-aligned station table."""
    if min_days < 1:
        raise ValueError("min_days must be positive")
    calendar = pd.date_range(cfg["data"]["start_date"], cfg["data"]["end_date"], freq="D")
    blocks = []
    for sid in cfg["data"]["stations"]:
        part = clean.loc[clean.station_id.astype(str).str.zfill(5).eq(sid), ["date", "tmax_c"]].copy()
        part["date"] = pd.to_datetime(part.date)
        if part.date.duplicated().any():
            raise ValueError("Duplicate station dates need to be audited before gap detection")
        aligned = part.set_index("date")["tmax_c"].reindex(calendar)
        missing = aligned.isna()
        group_id = missing.ne(missing.shift(fill_value=False)).cumsum()
        for _, group in missing.groupby(group_id):
            if bool(group.iloc[0]) and len(group) >= min_days:
                blocks.append({"station_id": sid, "start_date": group.index.min().date().isoformat(),
                               "end_date": group.index.max().date().isoformat(), "missing_days": len(group)})
    return pd.DataFrame(blocks, columns=["station_id", "start_date", "end_date", "missing_days"])
