from __future__ import annotations

import argparse
import json
from pathlib import Path

from perceptual_arbitration.trial_exports import export_trial_results


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export trial-level scores from completed checkpoints without fitting models."
    )
    parser.add_argument("--csv", default="data/data01_direction4priors.csv")
    parser.add_argument("--out", default="outputs/full_run", help="Completed analysis output directory")
    parser.add_argument("--scope", choices=["final", "oof", "both"], default="both")
    parser.add_argument("--include-subject", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace known export files")
    args = parser.parse_args()

    result = export_trial_results(
        _resolve(project_root, args.csv),
        _resolve(project_root, args.out),
        scope=args.scope,
        include_subject=args.include_subject,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": result["status"],
        "trial_export_dir": result["trial_export_dir"],
        "files": len(result["files"]),
        "usable_trials": result["usable_trials"],
    }, indent=2))


if __name__ == "__main__":
    main()
