from __future__ import annotations

import argparse
import json
from pathlib import Path

from perceptual_arbitration.data import load_direction_data
from perceptual_arbitration.posterior_predictive import run_checkpoint_posterior_predictive_checks
from perceptual_arbitration.publication import render_publication_report
from perceptual_arbitration.run_metadata import atomic_write_json, update_stage
from perceptual_arbitration.sp_posterior_predictive import run_lapse_excluded_posterior_predictive_checks


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run fit-independent posterior predictive checks from final checkpoints."
    )
    parser.add_argument("--csv", default="data/data01_direction4priors.csv")
    parser.add_argument("--out", default="outputs/full_run")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["static_hmm", "covariate_hmm"],
        default=["static_hmm", "covariate_hmm"],
    )
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sp-sensitivity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-report", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    csv_path = _resolve(project_root, args.csv)
    out_dir = _resolve(project_root, args.out)
    result = run_checkpoint_posterior_predictive_checks(
        csv_path,
        out_dir,
        models=args.models,
        n_simulations=args.n_simulations,
        seed=args.seed,
        resume=args.resume,
    )
    sp_result = None
    if args.sp_sensitivity:
        sp_result = run_lapse_excluded_posterior_predictive_checks(
            csv_path,
            out_dir,
            models=args.models,
            n_simulations=args.n_simulations,
            seed=args.seed,
            resume=args.resume,
        )

    report_result = None
    if args.render_report:
        report_result = render_publication_report(load_direction_data(csv_path), out_dir, package_html=True)
        manifest_path = out_dir / "run_manifest.json"
        parent = json.loads(manifest_path.read_text(encoding="utf-8"))
        update_stage(
            parent,
            "publication_report",
            "complete",
            report_status=report_result["status"],
            figures=len(report_result["figures"]),
            html=str(report_result["html_path"]) if report_result["html_path"] else None,
        )
        parent["publication_ready"] = bool(
            parent.get("publication_ready", False) and report_result["status"] == "ready"
        )
        parent.setdefault("validation", {})["report_issues"] = report_result["issues"]
        atomic_write_json(parent, manifest_path)

    print(json.dumps({
        "status": result["status"],
        "models": result["models"],
        "simulations_per_model": result["n_simulations"],
        "seed": result["seed"],
        "ppc_manifest": str((out_dir / "posterior_predictive_manifest.json").resolve()),
        "sp_sensitivity": bool(sp_result),
        "sp_manifest": str((out_dir / "posterior_predictive_sp_manifest.json").resolve()) if sp_result else None,
        "report": str(report_result["html_path"]) if report_result and report_result["html_path"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
