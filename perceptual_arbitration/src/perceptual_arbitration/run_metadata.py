from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)
    return path


def package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "joblib",
        "duckdb",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def new_run_manifest(
    config: dict,
    config_path: str | Path,
    csv_path: str | Path,
    out_dir: str | Path,
    n_jobs: int,
) -> dict[str, Any]:
    config_path = Path(config_path)
    csv_path = Path(csv_path)
    config_hash = stable_hash(config)
    data_hash = sha256_file(csv_path)
    run_key = stable_hash({"config_hash": config_hash, "data_hash": data_hash})
    return {
        "schema_version": 1,
        "status": "running",
        "publication_ready": False,
        "started_at": utc_now(),
        "completed_at": None,
        "run_key": run_key,
        "config_hash": config_hash,
        "data_sha256": data_hash,
        "config_path": str(config_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "out_dir": str(Path(out_dir).resolve()),
        "resolved_config": config,
        "execution": {"n_jobs": int(n_jobs)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "stages": {},
        "validation": {},
    }


def update_stage(manifest: dict[str, Any], stage: str, status: str, **details: Any) -> None:
    manifest.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_at": utc_now(),
        **details,
    }
