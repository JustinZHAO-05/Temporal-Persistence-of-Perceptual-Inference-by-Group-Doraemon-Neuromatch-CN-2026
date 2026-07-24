from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib


def atomic_joblib_dump(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)
    return path


def save_fit_checkpoint(path: str | Path, run_key: str, fit: Any) -> Path:
    return atomic_joblib_dump({"run_key": run_key, "fit": fit}, path)


def load_fit_checkpoint(path: str | Path, run_key: str) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("run_key") != run_key:
        return None
    return payload.get("fit")
