"""Hash data inputs and persist fitted estimators for a reproducible research run."""
from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

import joblib
import numpy
import pandas
import sklearn


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_provenance(cfg: dict, root: str | Path, *, demo: bool, station_metadata: pandas.DataFrame) -> dict:
    root = Path(root)
    input_files = sorted((root / "data" / "raw").glob("*.zip")) if not demo else []
    import yaml
    effective_config_bytes = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    return {
        "kind": "SYNTHETIC DEMO – NOT EMPIRICAL EVIDENCE" if demo else "DWD HISTORICAL ARCHIVES – RETROSPECTIVE",
        "source_name": "DWD historical daily KL" if not demo else "local deterministic synthetic generator",
        "git_sha": os.environ.get("GITHUB_SHA", "not recorded; local run"),
        "platform": platform.platform(), "python_version": platform.python_version(),
        "pandas": pandas.__version__, "numpy": numpy.__version__, "scikit_learn": sklearn.__version__,
        "station_ids": sorted(cfg["data"]["stations"]),
        "input_sha256": {str(path.relative_to(root)): sha256_file(path) for path in input_files},
        "observed_station_metadata_rows": int(len(station_metadata)),
        "effective_config_sha256": hashlib.sha256(effective_config_bytes).hexdigest(),
        "notes": "Archive content hashes and package versions allow detecting later source revisions; issue-time availability remains an unverified assumption.",
    }


def save_fitted_models(fitted: dict, directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, model in fitted.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Model name must be alphanumeric or underscores")
        path = directory / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        result[name] = str(path)
    return result
