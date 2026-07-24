from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from perceptual_arbitration.data import load_direction_data
from perceptual_arbitration.diagnostics import (
    bootstrap_model_differences,
    conditional_covariate_effect_intervals,
)
from perceptual_arbitration.model_selection import (
    empirical_bayes_summary,
    fit_final_models,
    fit_subject_level_hmms,
    run_core_cv,
    summarize_cv,
)
from perceptual_arbitration.publication import render_publication_report
from perceptual_arbitration.posterior_predictive import run_checkpoint_posterior_predictive_checks
from perceptual_arbitration.run_metadata import atomic_write_json, new_run_manifest, update_stage, utc_now
from perceptual_arbitration.sp_posterior_predictive import run_lapse_excluded_posterior_predictive_checks
from perceptual_arbitration.trial_exports import export_trial_results


def _resolve_project_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _ensure_csv_available(csv_path: Path, project_root: Path) -> None:
    if csv_path.exists():
        return
    candidates = [project_root.parent / csv_path.name, project_root.parent / "data01_direction4priors.csv"]
    for candidate in candidates:
        if candidate.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, csv_path)
            return
    raise FileNotFoundError(f"CSV not found at {csv_path} and no parent-directory copy was available")


def _bool_cfg(cfg: dict, key: str, default: bool) -> bool:
    return bool(cfg.get("diagnostics", {}).get(key, default))


def _write_manifest(manifest: dict, out_dir: Path) -> None:
    atomic_write_json(manifest, out_dir / "run_manifest.json")


def _selected_fits_converged(*frames: pd.DataFrame) -> tuple[bool, list[str]]:
    issues = []
    for label, frame in zip(["cross-validation", "final models", "subject fits"], frames):
        if frame.empty:
            issues.append(f"{label} results are empty")
        elif "converged" in frame and not frame["converged"].fillna(False).astype(bool).all():
            issues.append(f"one or more selected {label} fits are not converged")
    return not issues, issues


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--csv", default=None, help="Override CSV path")
    parser.add_argument("--out", default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--n-ppc-simulations", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse matching model checkpoints")
    parser.add_argument("--make-figures", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-report", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-trial-exports", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-sp-sensitivity", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config_path = _resolve_project_path(project_root, args.config)
    assert config_path is not None
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    csv_path = _resolve_project_path(project_root, args.csv or cfg["data"]["csv_path"])
    out_dir = _resolve_project_path(project_root, args.out or cfg["output"]["out_dir"])
    assert csv_path is not None and out_dir is not None
    _ensure_csv_available(csv_path, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_cfg = cfg.get("diagnostics", {})
    seed = int(args.seed if args.seed is not None else diag_cfg.get("seed", 42))
    n_bootstrap = int(args.n_bootstrap if args.n_bootstrap is not None else diag_cfg.get("n_bootstrap", 1000))
    n_ppc = int(args.n_ppc_simulations if args.n_ppc_simulations is not None else diag_cfg.get("n_ppc_simulations", 10))
    n_jobs = int(args.n_jobs if args.n_jobs is not None else diag_cfg.get("n_jobs", 1))
    make_figures = bool(args.make_figures if args.make_figures is not None else _bool_cfg(cfg, "make_figures", True))
    make_report = bool(args.make_report if args.make_report is not None else _bool_cfg(cfg, "make_report", True))
    make_trial_exports = bool(
        args.make_trial_exports
        if args.make_trial_exports is not None
        else _bool_cfg(cfg, "make_trial_exports", True)
    )
    make_sp_sensitivity = bool(
        args.make_sp_sensitivity
        if args.make_sp_sensitivity is not None
        else diag_cfg.get("make_sp_sensitivity", True)
    )

    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg.setdefault("diagnostics", {}).update({
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_ppc_simulations": n_ppc,
        "n_jobs": n_jobs,
        "make_figures": make_figures,
        "make_report": make_report,
        "make_trial_exports": make_trial_exports,
    })
    resolved_cfg["data"]["csv_path"] = str(csv_path.resolve())
    resolved_cfg["output"]["out_dir"] = str(out_dir.resolve())
    proposed_manifest = new_run_manifest(resolved_cfg, config_path, csv_path, out_dir, n_jobs)
    manifest_path = out_dir / "run_manifest.json"
    if args.resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_key") == proposed_manifest["run_key"]:
            proposed_manifest["started_at"] = existing.get("started_at", proposed_manifest["started_at"])
            proposed_manifest["stages"] = existing.get("stages", {})
    manifest = proposed_manifest
    _write_manifest(manifest, out_dir)
    run_key = manifest["run_key"]

    try:
        data = load_direction_data(csv_path)
        manifest["data"] = {
            "raw_rows": data.raw_n_rows,
            "usable_trials": len(data.df),
            "dropped_rows": data.dropped_n_rows,
            "sequences": len(data.sequences),
            "subjects": len(data.subject_values),
            "coherence_values": data.coh_values.tolist(),
            "prior_widths": data.prior_values.tolist(),
        }
        update_stage(manifest, "data", "complete")
        _write_manifest(manifest, out_dir)
        print(f"Loaded {len(data.df):,} trials, {len(data.sequences)} sequences, {len(data.subject_values)} subjects")
        print(f"Diagnostics seed={seed}; bootstrap={n_bootstrap}; PPC={n_ppc}; n_jobs={n_jobs}; resume={args.resume}")

        cv = run_core_cv(
            data,
            out_dir=out_dir,
            n_splits=int(cfg["cv"]["n_splits"]),
            n_restarts_hmm=int(cfg["training"]["n_restarts_hmm"]),
            n_restarts_ind=int(cfg["training"]["n_restarts_independent"]),
            n_restarts_serial=int(cfg["training"]["n_restarts_serial"]),
            n_restarts_iohmm=int(cfg["training"]["n_restarts_covariate_hmm"]),
            max_iter_em=int(cfg["training"]["max_iter_em"]),
            seed=seed,
            include_iohmm=bool(cfg["models"]["include_covariate_hmm"]),
            include_serial=bool(cfg["models"]["include_serial_dependence"]),
            tol_hmm=float(cfg["training"].get("tol_hmm", 1e-7)),
            n_jobs=n_jobs,
            resume=args.resume,
            run_key=run_key,
        )
        summary = summarize_cv(cv)
        summary.to_csv(out_dir / "cv_summary.csv", index=False)
        update_stage(manifest, "cross_validation", "complete", rows=len(cv))
        _write_manifest(manifest, out_dir)
        print("\nCross-validation summary:")
        print(summary.to_string(index=False))

        per_sequence = pd.read_csv(out_dir / "per_sequence_cv_results.csv")
        bootstrap = bootstrap_model_differences(per_sequence, n_bootstrap=n_bootstrap, seed=seed)
        bootstrap.to_csv(out_dir / "bootstrap_model_differences.csv", index=False)
        update_stage(manifest, "model_bootstrap", "complete", draws=n_bootstrap)
        _write_manifest(manifest, out_dir)

        final = fit_final_models(
            data,
            out_dir=out_dir,
            n_restarts_hmm=int(cfg["training"]["n_restarts_hmm"]),
            n_restarts_ind=int(cfg["training"]["n_restarts_independent"]),
            n_restarts_serial=int(cfg["training"]["n_restarts_serial"]),
            n_restarts_iohmm=int(cfg["training"]["n_restarts_covariate_hmm"]),
            max_iter_em=int(cfg["training"]["max_iter_em"]),
            include_iohmm=bool(cfg["models"]["include_covariate_hmm"]),
            include_serial=bool(cfg["models"]["include_serial_dependence"]),
            tol_hmm=float(cfg["training"].get("tol_hmm", 1e-7)),
            n_jobs=n_jobs,
            resume=args.resume,
            run_key=run_key,
        )
        update_stage(manifest, "final_models", "complete")
        _write_manifest(manifest, out_dir)

        if final["covariate"] is not None:
            covariate_intervals = conditional_covariate_effect_intervals(
                data,
                final["covariate"].params,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            covariate_intervals.to_csv(out_dir / "covariate_hmm_effect_intervals.csv", index=False)
            update_stage(manifest, "covariate_uncertainty", "complete", draws=n_bootstrap)
            _write_manifest(manifest, out_dir)

        if cfg["models"].get("include_subject_level", True):
            subject = fit_subject_level_hmms(
                data,
                out_dir=out_dir,
                n_restarts=int(cfg["training"]["n_restarts_subject"]),
                max_iter=int(cfg["training"]["max_iter_em"]),
                tol=float(cfg["training"].get("tol_hmm", 1e-7)),
                n_jobs=n_jobs,
                resume=args.resume,
                run_key=run_key,
            )
            empirical_bayes_summary(subject, out_dir=out_dir)
        else:
            subject = pd.DataFrame()
        update_stage(manifest, "subject_models", "complete", subjects=len(subject))
        _write_manifest(manifest, out_dir)

        if make_trial_exports:
            update_stage(manifest, "trial_exports", "running")
            _write_manifest(manifest, out_dir)
            trial_export = export_trial_results(
                csv_path,
                out_dir,
                scope="both",
                include_subject=bool(cfg["models"].get("include_subject_level", True)),
                overwrite=True,
                allow_running_manifest=True,
                update_parent_manifest=False,
            )
            update_stage(
                manifest,
                "trial_exports",
                "complete",
                files=len(trial_export["files"]),
                usable_trials=trial_export["usable_trials"],
                manifest=str((out_dir / "trial_exports" / "export_manifest.json").resolve()),
            )
            _write_manifest(manifest, out_dir)

        if n_ppc > 0:
            ppc_models = ["static_hmm"]
            if final["covariate"] is not None:
                ppc_models.append("covariate_hmm")
            update_stage(manifest, "posterior_predictive_checks", "running", models=ppc_models)
            _write_manifest(manifest, out_dir)
            ppc_result = run_checkpoint_posterior_predictive_checks(
                csv_path,
                out_dir,
                models=ppc_models,
                n_simulations=n_ppc,
                seed=seed,
                resume=args.resume,
                allow_running_manifest=True,
                update_parent_manifest=False,
            )
            update_stage(
                manifest,
                "posterior_predictive_checks",
                "complete",
                models=ppc_models,
                simulations_per_model=n_ppc,
                seed=seed,
                manifest=str((out_dir / "posterior_predictive_manifest.json").resolve()),
            )
            if make_sp_sensitivity:
                update_stage(
                    manifest,
                    "posterior_predictive_sp_sensitivity",
                    "running",
                    models=ppc_models,
                )
                _write_manifest(manifest, out_dir)
                sp_result = run_lapse_excluded_posterior_predictive_checks(
                    csv_path,
                    out_dir,
                    models=ppc_models,
                    n_simulations=n_ppc,
                    seed=seed,
                    resume=args.resume,
                    update_parent_manifest=False,
                )
                update_stage(
                    manifest,
                    "posterior_predictive_sp_sensitivity",
                    "complete",
                    models=ppc_models,
                    simulations_per_model=n_ppc,
                    seed=seed,
                    method=sp_result["state_path_summary"],
                    manifest=str((out_dir / "posterior_predictive_sp_manifest.json").resolve()),
                )
            else:
                update_stage(
                    manifest,
                    "posterior_predictive_sp_sensitivity",
                    "complete",
                    enabled=False,
                )
        else:
            update_stage(manifest, "posterior_predictive_checks", "complete", models=[], simulations_per_model=0)
            update_stage(
                manifest,
                "posterior_predictive_sp_sensitivity",
                "complete",
                enabled=False,
            )

        restart_frames = []
        for name in ["restart_diagnostics_cv.csv", "restart_diagnostics_final.csv", "restart_diagnostics_subject.csv"]:
            path = out_dir / name
            if path.exists():
                restart_frames.append(pd.read_csv(path))
        if restart_frames:
            pd.concat(restart_frames, ignore_index=True).to_csv(out_dir / "restart_diagnostics.csv", index=False)

        info = pd.read_csv(out_dir / "model_info_criteria.csv")
        converged, convergence_issues = _selected_fits_converged(cv, info, subject)
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        manifest["publication_ready"] = converged
        manifest["validation"] = {"selected_fits_converged": converged, "issues": convergence_issues}
        _write_manifest(manifest, out_dir)

        if make_figures or make_report:
            report = render_publication_report(data, out_dir, package_html=make_report)
            update_stage(
                manifest,
                "publication_report",
                "complete",
                report_status=report["status"],
                figures=len(report["figures"]),
                html=str(report["html_path"]) if report["html_path"] else None,
            )
            manifest["publication_ready"] = manifest["publication_ready"] and report["status"] == "ready"
            manifest["validation"]["report_issues"] = report["issues"]
            _write_manifest(manifest, out_dir)
            print(f"\nPublication report status: {report['status']}; figures: {len(report['figures'])}")
            if report["html_path"]:
                print(f"HTML report: {report['html_path']}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["publication_ready"] = False
        manifest.setdefault("validation", {})["fatal_error"] = repr(exc)
        _write_manifest(manifest, out_dir)
        raise


if __name__ == "__main__":
    main()
