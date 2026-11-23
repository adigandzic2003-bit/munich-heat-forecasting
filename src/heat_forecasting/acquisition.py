"""Download and parse immutable DWD daily climate archives."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

ARCHIVE_PATTERN = re.compile(r'tageswerte_KL_(\d{5})_(\d{8})_(\d{8})_hist\.zip')
DWD_FIELDS = {
    "STATIONS_ID": "station_id", "MESS_DATUM": "date", "TXK": "tmax_c",
    "TMK": "tmean_c", "TNK": "tmin_c", "RSK": "rain_mm", "SDK": "sun_hours",
    "UPM": "humidity_pct", "QN_4": "quality_flag_temp", "QN_3": "quality_flag_rain",
}


def archive_catalog(index_html: str, base_url: str) -> dict[str, str]:
    """Choose the archive with the latest end date for each station ID."""
    found: dict[str, tuple[str, str]] = {}
    for match in ARCHIVE_PATTERN.finditer(index_html):
        sid, _start, end = match.groups()
        if sid not in found or end > found[sid][0]:
            found[sid] = (end, urljoin(base_url, match.group(0)))
    if not found:
        raise ValueError("No DWD KL historical zip archives found in directory index")
    return {sid: item[1] for sid, item in found.items()}


def parse_station_zip(archive: bytes, expected_station: str) -> pd.DataFrame:
    """Load the DWD semicolon-delimited daily measurements inside a zip archive."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        candidates = [n for n in zipped.namelist() if re.search(r'produkt_klima_tag_.*\.txt$', n, re.I)]
        if len(candidates) != 1:
            raise ValueError(f"Expected one daily climate text file, found {candidates}")
        with zipped.open(candidates[0]) as handle:
            frame = pd.read_csv(handle, sep=";", encoding="latin-1", low_memory=False)
    frame.columns = frame.columns.str.strip()
    if not {"STATIONS_ID", "MESS_DATUM", "TXK"}.issubset(frame.columns):
        raise ValueError("DWD station archive missing required STATIONS_ID / MESS_DATUM / TXK")
    frame = frame.rename(columns=DWD_FIELDS)
    frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
    if set(frame["station_id"]) != {expected_station}:
        raise ValueError(f"Archive station ID does not match expected {expected_station}")
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="raise")
    numeric = [c for c in DWD_FIELDS.values() if c not in ("station_id", "date") and c in frame]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace(-999, np.nan)
    return frame[["date", "station_id", *numeric]].sort_values("date").reset_index(drop=True)


def fetch_historical(cfg: dict, target_dir: str | Path, *, session: requests.Session | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch station archives; original ZIP bytes are cached read-only in data/raw/."""
    base_url = cfg["data"]["source"].rstrip("/") + "/"
    raw_dir = Path(target_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    response = client.get(base_url, timeout=45)
    response.raise_for_status()
    catalog = archive_catalog(response.text, base_url)
    frames = []
    provenance = []
    for sid, metadata in cfg["data"]["stations"].items():
        if sid not in catalog:
            raise RuntimeError(f"Station {sid} ({metadata['name']}) absent from historical archive listing")
        url = catalog[sid]
        cache_file = raw_dir / Path(url).name
        if not cache_file.is_file():
            download = client.get(url, timeout=120)
            download.raise_for_status()
            if not zipfile.is_zipfile(io.BytesIO(download.content)):
                raise ValueError(f"DWD response for {sid} was not a valid ZIP")
            cache_file.write_bytes(download.content)
        frames.append(parse_station_zip(cache_file.read_bytes(), sid))
        provenance.append({"station_id": sid, **metadata, "source_url": url, "cached_file": str(cache_file), "archive_bytes": cache_file.stat().st_size})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(provenance)
