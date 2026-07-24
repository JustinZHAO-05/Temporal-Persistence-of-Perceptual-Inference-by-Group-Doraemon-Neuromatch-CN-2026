from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .checkpoints import load_fit_checkpoint
from .circular import circ_diff
from .covariate_hmm import (
    forward_backward_timevarying,
    transition_logA_sequence,
)
from .data import DataBundle, load_direction_data
from .diagnostics import (
    posterior_predictive_summaries,
    ppc_histogram_intervals,
    ppc_metric_intervals,
    run_length_calibration,
    simulate_covariate_hmm_draw,
    simulate_static_hmm_draw,
    state_run_lengths_from_posterior,
    state_run_lengths_from_simulation,
    summarize_run_lengths,
)
from .hmm import HMMParams, STATE_NAMES, emission_logB_stable
from .run_metadata import atomic_write_json, sha256_file, stable_hash, utc_now


MODEL_SPECS = {
    "static_hmm": {
        "label": "HMM_static",
        "checkpoint": "final_hmm_static.joblib",
        "prefix": "",
    },
    "covariate_hmm": {
        "label": "Covariate_HMM",
        "checkpoint": "final_covariate_hmm.joblib",
        "prefix": "covariate_hmm_",
    },
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        na_rep="",
        float_format="%.17g",
        lineterminator="\n",
    )
    os.replace(temporary, path)
    return path


def _concatenate_csv_shards(shards: list[Path], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for index, shard in enumerate(shards):
            with shard.open("rb") as source:
                if index:
                    source.readline()
                shutil.copyfileobj(source, output, length=1024 * 1024)
    os.replace(temporary, destination)
    return destination


def _file_record(path: Path, out_dir: Path, rows: int | None = None, columns: int | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(out_dir).as_posix(),
        "rows": rows,
        "columns": columns,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _covariate_posterior(data: DataBundle, params: Any) -> pd.DataFrame:
    rows = []
    hmm_like = HMMParams(
        pi=params.pi,
        A=np.ones((3, 3), dtype=float) / 3.0,
        kappa_s=params.kappa_s,
        kappa_p=params.kappa_p,
    )
    for sid, idx in enumerate(data.sequences):
        log_b = emission_logB_stable(data, hmm_like, idx)
        log_a = transition_logA_sequence(params, data.X_transition[idx])
        _, gamma, _ = forward_backward_timevarying(
            log_b,
            np.log(np.maximum(params.pi, 1e-300)),
            log_a,
        )
        for position, row_idx_value in enumerate(idx):
            row_idx = int(row_idx_value)
            source = data.df.loc[row_idx]
            rows.append({
                "row_index": row_idx,
                "seq_id": sid,
                "subject_id": source["subject_id"],
                "session_id": source["session_id"],
                "run_id": source["run_id"],
                "trial_index": source["trial_index"],
                "p_sensory": gamma[position, 0],
                "p_prior": gamma[position, 1],
                "p_lapse": gamma[position, 2],
                "state_map": STATE_NAMES[int(np.argmax(gamma[position]))],
                "state_map_method": "smoothed_marginal_argmax_not_viterbi",
            })
    return pd.DataFrame(rows)


def _validate_draw(draw: pd.DataFrame, data: DataBundle, model: str, simulation: int) -> dict[str, Any]:
    if len(draw) != len(data.df):
        raise RuntimeError(f"{model} simulation {simulation} has {len(draw)} rows, expected {len(data.df)}")
    if draw["row_index"].nunique() != len(data.df) or set(draw["simulation"].unique()) != {simulation}:
        raise RuntimeError(f"{model} simulation {simulation} has invalid trial or simulation keys")
    if draw["seq_id"].nunique() != len(data.sequences):
        raise RuntimeError(f"{model} simulation {simulation} does not contain all sequences")
    if not np.isfinite(draw["estimate_deg"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{model} simulation {simulation} contains a non-finite response")
    starts = draw["is_sequence_start"].astype(bool)
    if draw.loc[starts, ["transition_probability_used", "transition_probability_row_sum"]].notna().any().any():
        raise RuntimeError("Sequence-start transition fields must be blank")
    nonstarts = ~starts
    normalization_error = float(np.max(np.abs(draw.loc[nonstarts, "transition_probability_row_sum"] - 1.0)))
    if normalization_error > 1e-12:
        raise RuntimeError(f"Transition probability rows do not normalize: {normalization_error}")
    prev_error_error = 0.0
    if model == "covariate_hmm":
        estimate = np.deg2rad(draw["estimate_deg"].to_numpy(dtype=float))
        theta = data.theta
        recorded = draw["transition_prev_error_deg"].to_numpy(dtype=float)
        for idx in data.sequences:
            expected = np.abs(np.rad2deg(circ_diff(estimate[idx[:-1]], theta[idx[:-1]])))
            if len(expected):
                prev_error_error = max(
                    prev_error_error,
                    float(np.max(np.abs(recorded[idx[1:]] - expected))),
                )
        if prev_error_error > 1e-10:
            raise RuntimeError(f"Simulated previous-error covariates are inconsistent: {prev_error_error}")
    return {
        "rows": len(draw),
        "sequences": int(draw["seq_id"].nunique()),
        "max_transition_row_sum_error": normalization_error,
        "max_simulated_previous_error_error_deg": prev_error_error,
    }


def _coverage_table(metric_intervals: pd.DataFrame, model_label: str) -> pd.DataFrame:
    detailed = metric_intervals.copy()
    detailed["covered"] = (
        (detailed["observed"] >= detailed["simulated_ci_low"])
        & (detailed["observed"] <= detailed["simulated_ci_high"])
    )
    coverage = detailed.groupby("metric", as_index=False).agg(
        cells=("covered", "size"),
        covered=("covered", "sum"),
    )
    coverage["coverage_rate"] = coverage["covered"] / coverage["cells"]
    coverage.insert(0, "model", model_label)
    return coverage


def _model_paths(out_dir: Path, model: str) -> dict[str, Path]:
    prefix = str(MODEL_SPECS[model]["prefix"])
    return {
        "responses": out_dir / f"{prefix}posterior_predictive_simulated_responses.csv",
        "summary": out_dir / f"{prefix}posterior_predictive_condition_summary.csv",
        "histograms": out_dir / f"{prefix}posterior_predictive_histograms.csv",
        "run_lengths": out_dir / f"{prefix}state_run_lengths.csv",
        "run_summary": out_dir / f"{prefix}state_run_length_summary.csv",
        "metric_intervals": out_dir / f"{prefix}posterior_predictive_metric_intervals.csv",
        "histogram_intervals": out_dir / f"{prefix}posterior_predictive_histogram_intervals.csv",
        "run_calibration": out_dir / f"{prefix}state_run_length_calibration.csv",
        "coverage": out_dir / f"{prefix}posterior_predictive_coverage.csv",
    }


def _valid_completed_model(progress: dict[str, Any], model: str, out_dir: Path) -> bool:
    entry = progress.get("models", {}).get(model, {})
    if entry.get("status") != "complete":
        return False
    for record in entry.get("files", []):
        path = out_dir / record["path"]
        if not path.exists() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            return False
    return True


def _valid_completed_manifest(manifest: dict[str, Any], ppc_key: str, out_dir: Path) -> bool:
    if manifest.get("status") != "complete" or manifest.get("ppc_key") != ppc_key:
        return False
    for record in manifest.get("files", []):
        path = out_dir / str(record["path"])
        if (
            not path.exists()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            return False
    return True


def _run_model(
    data: DataBundle,
    fit: Any,
    model: str,
    out_dir: Path,
    work_dir: Path,
    n_simulations: int,
    seed: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label = str(MODEL_SPECS[model]["label"])
    model_work = work_dir / model
    model_work.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    summary_parts: list[pd.DataFrame] = []
    histogram_parts: list[pd.DataFrame] = []
    run_parts: list[pd.DataFrame] = []
    observed_summary: pd.DataFrame | None = None
    observed_histograms: pd.DataFrame | None = None
    validation_draws: list[dict[str, Any]] = []

    for simulation in range(n_simulations):
        shard = model_work / f"draw_{simulation:04d}.csv"
        summary_shard = model_work / f"draw_{simulation:04d}_summary.csv"
        histogram_shard = model_work / f"draw_{simulation:04d}_histograms.csv"
        run_shard = model_work / f"draw_{simulation:04d}_runs.csv"
        ready = resume and all(path.exists() and path.stat().st_size > 0 for path in [shard, summary_shard, histogram_shard, run_shard])
        if ready:
            draw = pd.read_csv(shard)
            summary = pd.read_csv(summary_shard)
            histograms = pd.read_csv(histogram_shard)
            runs = pd.read_csv(run_shard)
        else:
            draw = (
                simulate_static_hmm_draw(data, fit.params, simulation=simulation, seed=seed)
                if model == "static_hmm"
                else simulate_covariate_hmm_draw(data, fit.params, simulation=simulation, seed=seed)
            )
            summary_all, histogram_all = posterior_predictive_summaries(data, draw)
            summary = summary_all[summary_all["source"] == "simulated"].copy()
            histograms = histogram_all[histogram_all["source"] == "simulated"].copy()
            runs = state_run_lengths_from_simulation(draw)
            _atomic_csv(draw, shard)
            _atomic_csv(summary, summary_shard)
            _atomic_csv(histograms, histogram_shard)
            _atomic_csv(runs, run_shard)
        validation_draws.append(_validate_draw(draw, data, model, simulation))
        if observed_summary is None or observed_histograms is None:
            summary_all, histogram_all = posterior_predictive_summaries(data, draw.iloc[0:0].copy())
            observed_summary = summary_all[summary_all["source"] == "observed"].copy()
            observed_histograms = histogram_all[histogram_all["source"] == "observed"].copy()
        shards.append(shard)
        summary_parts.append(summary)
        histogram_parts.append(histograms)
        run_parts.append(runs)
        print(f"PPC {model}: completed draw {simulation + 1}/{n_simulations}", flush=True)

    assert observed_summary is not None and observed_histograms is not None
    paths = _model_paths(out_dir, model)
    _concatenate_csv_shards(shards, paths["responses"])
    summary = pd.concat([observed_summary, *summary_parts], ignore_index=True)
    histograms = pd.concat([observed_histograms, *histogram_parts], ignore_index=True)

    if model == "static_hmm":
        posterior = pd.read_csv(out_dir / "posterior_states.csv")
    else:
        posterior = _covariate_posterior(data, fit.params)
        _atomic_csv(posterior, out_dir / "covariate_hmm_posterior_states.csv")
    observed_runs = state_run_lengths_from_posterior(posterior)
    run_lengths = pd.concat([observed_runs, *run_parts], ignore_index=True)
    metric_intervals = ppc_metric_intervals(summary)
    histogram_intervals = ppc_histogram_intervals(histograms)
    run_calibration = run_length_calibration(run_lengths)
    metric_intervals["n_simulations"] = n_simulations
    histogram_intervals["n_simulations"] = n_simulations
    run_calibration["n_simulations"] = n_simulations
    coverage = _coverage_table(metric_intervals, label)
    for frame in [summary, histograms, run_lengths, metric_intervals, histogram_intervals, run_calibration]:
        frame.insert(0, "model", label)

    frames = {
        "summary": summary,
        "histograms": histograms,
        "run_lengths": run_lengths,
        "run_summary": summarize_run_lengths(run_lengths).assign(model=label),
        "metric_intervals": metric_intervals,
        "histogram_intervals": histogram_intervals,
        "run_calibration": run_calibration,
        "coverage": coverage,
    }
    records = [_file_record(paths["responses"], out_dir, n_simulations * len(data.df), 19)]
    for key, frame in frames.items():
        path = _atomic_csv(frame, paths[key])
        records.append(_file_record(path, out_dir, len(frame), len(frame.columns)))

    validation = {
        "draws": n_simulations,
        "rows_per_draw": len(data.df),
        "total_simulated_rows": n_simulations * len(data.df),
        "sequences_per_draw": len(data.sequences),
        "simulation_ids": [0, n_simulations - 1],
        "max_transition_row_sum_error": max(row["max_transition_row_sum_error"] for row in validation_draws),
        "max_simulated_previous_error_error_deg": max(row["max_simulated_previous_error_error_deg"] for row in validation_draws),
    }
    shutil.rmtree(model_work)
    return records, validation


def run_checkpoint_posterior_predictive_checks(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    models: Iterable[str] = ("static_hmm", "covariate_hmm"),
    n_simulations: int = 100,
    seed: int = 42,
    resume: bool = True,
    allow_running_manifest: bool = False,
    update_parent_manifest: bool = True,
) -> dict[str, Any]:
    """Run deterministic PPCs from completed checkpoints without fitting models."""
    csv_path = Path(csv_path).resolve()
    out_dir = Path(out_dir).resolve()
    model_list = list(dict.fromkeys(models))
    unknown = [model for model in model_list if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown PPC models: {unknown}")
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    manifest_path = out_dir / "run_manifest.json"
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parent.get("status") != "complete" and not allow_running_manifest:
        raise RuntimeError("PPC execution requires a complete parent run manifest")
    if sha256_file(csv_path) != parent.get("data_sha256"):
        raise RuntimeError("Dataset hash does not match the fitted run")
    run_key = str(parent["run_key"])
    data = load_direction_data(csv_path)
    fits: dict[str, Any] = {}
    checkpoint_records: dict[str, dict[str, Any]] = {}
    for model in model_list:
        checkpoint = out_dir / "checkpoints" / str(MODEL_SPECS[model]["checkpoint"])
        fit = load_fit_checkpoint(checkpoint, run_key)
        if fit is None:
            raise RuntimeError(f"Missing or incompatible checkpoint: {checkpoint}")
        fits[model] = fit
        checkpoint_records[model] = {
            "path": checkpoint.relative_to(out_dir).as_posix(),
            "sha256_before": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "seed": int(fit.seed),
            "converged": bool(fit.converged),
        }

    ppc_key = stable_hash({
        "run_key": run_key,
        "data_sha256": parent["data_sha256"],
        "models": model_list,
        "n_simulations": int(n_simulations),
        "seed": int(seed),
        "checkpoint_hashes": {model: row["sha256_before"] for model, row in checkpoint_records.items()},
        "schema_version": 1,
    })
    completed_manifest_path = out_dir / "posterior_predictive_manifest.json"
    if resume and completed_manifest_path.exists():
        completed_manifest = json.loads(completed_manifest_path.read_text(encoding="utf-8"))
        if _valid_completed_manifest(completed_manifest, ppc_key, out_dir):
            for model, row in checkpoint_records.items():
                recorded = completed_manifest.get("checkpoints", {}).get(model, {})
                if recorded.get("sha256_after") != row["sha256_before"]:
                    raise RuntimeError(f"Completed PPC checkpoint provenance changed: {model}")
            if update_parent_manifest:
                configured = int(parent.get("resolved_config", {}).get("diagnostics", {}).get("n_ppc_simulations", n_simulations))
                parent.setdefault("stages", {})["posterior_predictive_checks"] = {
                    "status": "complete",
                    "updated_at": utc_now(),
                    "models": model_list,
                    "simulations_per_model": int(n_simulations),
                    "configured_simulations": configured,
                    "execution_override": configured != int(n_simulations),
                    "seed": int(seed),
                    "manifest": str(completed_manifest_path.resolve()),
                }
                atomic_write_json(parent, manifest_path)
            print("PPC: reusing validated completed outputs", flush=True)
            return completed_manifest
    work_dir = out_dir / ".ppc_work" / ppc_key
    progress_path = work_dir / "progress.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {"ppc_key": ppc_key, "models": {}}
        atomic_write_json(progress, progress_path)

    all_files: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for model in model_list:
        if resume and _valid_completed_model(progress, model, out_dir):
            records = progress["models"][model]["files"]
            model_validation = progress["models"][model]["validation"]
            print(f"PPC {model}: reusing validated completed outputs", flush=True)
        else:
            records, model_validation = _run_model(
                data,
                fits[model],
                model,
                out_dir,
                work_dir,
                int(n_simulations),
                int(seed),
                resume,
            )
            progress.setdefault("models", {})[model] = {
                "status": "complete",
                "files": records,
                "validation": model_validation,
            }
            atomic_write_json(progress, progress_path)
        all_files.extend(records)
        validation[model] = model_validation

    metric_frames = [pd.read_csv(_model_paths(out_dir, model)["metric_intervals"]) for model in model_list]
    histogram_frames = [pd.read_csv(_model_paths(out_dir, model)["histogram_intervals"]) for model in model_list]
    run_frames = [pd.read_csv(_model_paths(out_dir, model)["run_calibration"]) for model in model_list]
    coverage_frames = [pd.read_csv(_model_paths(out_dir, model)["coverage"]) for model in model_list]
    combined = {
        "posterior_predictive_model_metric_intervals.csv": pd.concat(metric_frames, ignore_index=True),
        "posterior_predictive_model_histogram_intervals.csv": pd.concat(histogram_frames, ignore_index=True),
        "posterior_predictive_model_run_length_calibration.csv": pd.concat(run_frames, ignore_index=True),
        "posterior_predictive_model_coverage.csv": pd.concat(coverage_frames, ignore_index=True),
    }
    for name, frame in combined.items():
        path = _atomic_csv(frame, out_dir / name)
        all_files.append(_file_record(path, out_dir, len(frame), len(frame.columns)))

    for model, row in checkpoint_records.items():
        checkpoint = out_dir / row["path"]
        row["sha256_after"] = sha256_file(checkpoint)
        row["unchanged"] = row["sha256_after"] == row["sha256_before"]
        if not row["unchanged"]:
            raise RuntimeError(f"Fit checkpoint changed during PPC execution: {model}")

    configured = int(parent.get("resolved_config", {}).get("diagnostics", {}).get("n_ppc_simulations", n_simulations))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": utc_now(),
        "fit_independent": True,
        "ppc_key": ppc_key,
        "parent_run_key": run_key,
        "data_sha256": parent["data_sha256"],
        "models": model_list,
        "seed": int(seed),
        "n_simulations": int(n_simulations),
        "configured_n_simulations": configured,
        "execution_override": configured != int(n_simulations),
        "conditioning": "observed_design_with_simulated_states_responses_and_recursive_previous_error",
        "state_path_summary": "smoothed_marginal_argmax_not_viterbi",
        "transition_scaler": "final_all_data_scaler_reconstructed_from_fitted_dataset",
        "checkpoints": checkpoint_records,
        "files": sorted(all_files, key=lambda row: row["path"]),
        "validation": validation,
    }
    atomic_write_json(manifest, out_dir / "posterior_predictive_manifest.json")

    if update_parent_manifest:
        parent.setdefault("stages", {})["posterior_predictive_checks"] = {
            "status": "complete",
            "updated_at": utc_now(),
            "models": model_list,
            "simulations_per_model": int(n_simulations),
            "configured_simulations": configured,
            "execution_override": configured != int(n_simulations),
            "seed": int(seed),
            "manifest": str((out_dir / "posterior_predictive_manifest.json").resolve()),
        }
        atomic_write_json(parent, manifest_path)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    return manifest
