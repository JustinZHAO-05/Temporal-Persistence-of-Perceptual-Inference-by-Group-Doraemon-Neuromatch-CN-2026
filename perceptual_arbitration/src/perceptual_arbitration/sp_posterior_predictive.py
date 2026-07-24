from __future__ import annotations

from dataclasses import replace
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .checkpoints import load_fit_checkpoint
from .circular import circ_absdiff, circ_diff
from .covariate_hmm import (
    CovariateHMMParams,
    transition_logA_sequence,
)
from .data import DataBundle, load_direction_data
from .diagnostics import _condition_metrics, ppc_histogram_intervals, ppc_metric_intervals
from .hmm import HMMParams, STATE_NAMES, emission_logB_stable
from .posterior_predictive import MODEL_SPECS, _atomic_csv, _file_record, _model_paths
from .run_metadata import atomic_write_json, sha256_file, stable_hash, utc_now


SP_METHOD = "smoothed_marginal_argmax_not_viterbi"
SP_SCHEMA_VERSION = 1
SP_METRICS = [
    "mean_abs_error_deg",
    "median_abs_error_deg",
    "mean_cos_error",
    "circular_std_error_deg",
    "prior_like_rate",
]


def _smoothed_gamma_only(
    log_b: np.ndarray,
    initial_probability: np.ndarray,
    transition_probability: np.ndarray,
) -> np.ndarray:
    """Scaled forward-backward marginals without unused transition posteriors."""
    log_b = np.asarray(log_b, dtype=float)
    initial_probability = np.asarray(initial_probability, dtype=float)
    transition_probability = np.asarray(transition_probability, dtype=float)
    n_trials, n_states = log_b.shape
    if transition_probability.ndim == 2:
        if transition_probability.shape != (n_states, n_states):
            raise ValueError("Static transition matrix has the wrong shape")
    elif transition_probability.shape != (max(n_trials - 1, 0), n_states, n_states):
        raise ValueError("Time-varying transition matrices have the wrong shape")

    emission = np.exp(log_b - np.max(log_b, axis=1, keepdims=True))
    alpha = np.empty_like(emission)
    alpha[0] = initial_probability * emission[0]
    alpha[0] /= max(float(alpha[0].sum()), 1e-300)
    for trial in range(1, n_trials):
        transition = transition_probability if transition_probability.ndim == 2 else transition_probability[trial - 1]
        alpha[trial] = (alpha[trial - 1] @ transition) * emission[trial]
        alpha[trial] /= max(float(alpha[trial].sum()), 1e-300)

    beta = np.empty_like(emission)
    beta[-1] = 1.0
    for trial in range(n_trials - 2, -1, -1):
        transition = transition_probability if transition_probability.ndim == 2 else transition_probability[trial]
        beta[trial] = transition @ (emission[trial + 1] * beta[trial + 1])
        beta[trial] /= max(float(beta[trial].sum()), 1e-300)

    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    return gamma


def _response_bundle(
    data: DataBundle,
    response_deg: np.ndarray,
    *,
    covariate: bool,
) -> tuple[DataBundle, np.ndarray, float]:
    response_deg = np.asarray(response_deg, dtype=float)
    if response_deg.shape != (len(data.df),) or not np.isfinite(response_deg).all():
        raise ValueError("Decoded responses must be one finite value per analyzed trial")
    y = np.deg2rad(np.mod(response_deg, 360.0))
    cos_s = np.cos(circ_diff(y, data.theta))
    cos_p = np.cos(circ_diff(y, data.prior_mu))
    x_transition = data.X_transition
    expected_prev_error = np.full(len(data.df), np.nan, dtype=float)
    if covariate:
        raw = data.X_transition_raw.copy()
        raw_index = data.transition_names.index("prev_error") - 1
        for idx in data.sequences:
            if len(idx) > 1:
                values = circ_absdiff(y[idx[:-1]], data.theta[idx[:-1]])
                raw[idx[1:], raw_index] = values
                expected_prev_error[idx[1:]] = values
        x_transition = np.column_stack([
            np.ones(len(data.df), dtype=float),
            (raw - data.transition_means) / data.transition_sds,
        ])
    bundle = replace(
        data,
        y=y,
        cos_s=cos_s,
        cos_p=cos_p,
        X_transition=x_transition,
    )
    return bundle, expected_prev_error, float(np.max(np.abs(bundle.cos_s)) if len(bundle.cos_s) else 0.0)


def decode_smoothed_marginal_states(
    data: DataBundle,
    params: HMMParams | CovariateHMMParams,
    response_deg: np.ndarray,
    *,
    model: str,
    recorded_prev_error_deg: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Decode one complete response dataset with forward-backward marginals."""
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown PPC model: {model}")
    covariate = model == "covariate_hmm"
    bundle, expected_prev_error, _ = _response_bundle(data, response_deg, covariate=covariate)
    gamma_all = np.empty((len(data.df), len(STATE_NAMES)), dtype=float)
    if covariate:
        assert isinstance(params, CovariateHMMParams)
        hmm_like = HMMParams(
            pi=params.pi,
            A=np.ones((3, 3), dtype=float) / 3.0,
            kappa_s=params.kappa_s,
            kappa_p=params.kappa_p,
        )
    else:
        assert isinstance(params, HMMParams)
        hmm_like = params

    for idx in data.sequences:
        log_b = emission_logB_stable(bundle, hmm_like, idx)
        if covariate:
            log_a = transition_logA_sequence(params, bundle.X_transition[idx])
            gamma = _smoothed_gamma_only(log_b, params.pi, np.exp(log_a))
        else:
            gamma = _smoothed_gamma_only(log_b, params.pi, params.A)
        gamma_all[idx] = gamma

    row_sums = gamma_all.sum(axis=1)
    labels = np.asarray(STATE_NAMES, dtype=object)[np.argmax(gamma_all, axis=1)]
    decoded = pd.DataFrame({
        "row_index": np.arange(len(data.df), dtype=int),
        "p_sensory": gamma_all[:, 0],
        "p_prior": gamma_all[:, 1],
        "p_lapse": gamma_all[:, 2],
        "state_map": labels,
        "state_map_method": SP_METHOD,
    })
    prev_error_error = 0.0
    if covariate and recorded_prev_error_deg is not None:
        recorded = np.deg2rad(np.asarray(recorded_prev_error_deg, dtype=float))
        nonstart = np.isfinite(expected_prev_error)
        if recorded.shape != expected_prev_error.shape or not np.isfinite(recorded[nonstart]).all():
            raise ValueError("Recorded simulated previous errors are incomplete")
        prev_error_error = float(np.max(np.abs(recorded[nonstart] - expected_prev_error[nonstart])))
    validation = {
        "max_posterior_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
        "max_simulated_previous_error_error_rad": prev_error_error,
    }
    return decoded, validation


def _conditional_summaries(
    data: DataBundle,
    response_deg: np.ndarray,
    keep: np.ndarray,
    *,
    source: str,
    simulation: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = np.asarray(keep, dtype=bool)
    response_deg = np.asarray(response_deg, dtype=float)
    if keep.shape != (len(data.df),) or response_deg.shape != (len(data.df),):
        raise ValueError("Conditional PPC arrays do not match the dataset")
    expected_conditions = set(
        map(tuple, data.df[["motion_coherence", "prior_std"]].drop_duplicates().to_numpy())
    )
    rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    bins = np.arange(-180, 181, 15)
    retained = data.df.loc[keep, ["motion_coherence", "prior_std", "motion_direction", "prior_mean"]].copy()
    retained["estimate_deg"] = response_deg[keep]
    actual_conditions = set(
        map(tuple, retained[["motion_coherence", "prior_std"]].drop_duplicates().to_numpy())
    )
    if actual_conditions != expected_conditions:
        missing = sorted(expected_conditions - actual_conditions)
        raise RuntimeError(f"Lapse exclusion emptied condition cells: {missing}")
    for (coherence, prior_std), group in retained.groupby(["motion_coherence", "prior_std"], sort=True):
        y = np.deg2rad(np.mod(group["estimate_deg"].to_numpy(dtype=float), 360.0))
        theta = np.deg2rad(np.mod(group["motion_direction"].to_numpy(dtype=float), 360.0))
        prior = np.deg2rad(np.mod(group["prior_mean"].to_numpy(dtype=float), 360.0))
        rows.append({
            "source": source,
            "simulation": int(simulation),
            "motion_coherence": float(coherence),
            "prior_std": float(prior_std),
            **_condition_metrics(group, y, theta, prior),
        })
        for reference, center in [("stimulus", theta), ("prior", prior)]:
            relative = np.rad2deg(circ_diff(y, center))
            counts, edges = np.histogram(relative, bins=bins)
            denominator = max(int(counts.sum()), 1)
            for count, left, right in zip(counts, edges[:-1], edges[1:]):
                histogram_rows.append({
                    "source": source,
                    "simulation": int(simulation),
                    "motion_coherence": float(coherence),
                    "prior_std": float(prior_std),
                    "reference": reference,
                    "bin_left_deg": float(left),
                    "bin_right_deg": float(right),
                    "count": int(count),
                    "proportion": float(count / denominator),
                })
    return pd.DataFrame(rows), pd.DataFrame(histogram_rows)


def _retention_table(
    data: DataBundle,
    decoded_state: np.ndarray,
    *,
    source: str,
    simulation: int,
) -> pd.DataFrame:
    frame = data.df[["motion_coherence", "prior_std"]].copy()
    frame["decoded_state"] = np.asarray(decoded_state, dtype=object)
    grouped = frame.groupby(
        ["motion_coherence", "prior_std", "decoded_state"], sort=True
    ).size().rename("n").reset_index()
    complete = pd.MultiIndex.from_product([
        sorted(frame["motion_coherence"].unique()),
        sorted(frame["prior_std"].unique()),
        STATE_NAMES,
    ], names=["motion_coherence", "prior_std", "decoded_state"]).to_frame(index=False)
    result = complete.merge(grouped, how="left").fillna({"n": 0})
    result["n"] = result["n"].astype(int)
    result["condition_n"] = result.groupby(["motion_coherence", "prior_std"])["n"].transform("sum")
    result["fraction_of_condition"] = result["n"] / result["condition_n"]
    result["retained"] = result["decoded_state"] != "L_lapse"
    result.insert(0, "simulation", int(simulation))
    result.insert(0, "source", source)
    return result


def _confusion_table(
    generating_state: np.ndarray,
    decoded_state: np.ndarray,
    simulation: int,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "generating_state": np.asarray(generating_state, dtype=object),
        "decoded_state": np.asarray(decoded_state, dtype=object),
    })
    grouped = frame.groupby(["generating_state", "decoded_state"], sort=True).size().rename("n").reset_index()
    complete = pd.MultiIndex.from_product(
        [STATE_NAMES, STATE_NAMES], names=["generating_state", "decoded_state"]
    ).to_frame(index=False)
    result = complete.merge(grouped, how="left").fillna({"n": 0})
    result["n"] = result["n"].astype(int)
    result["generating_state_n"] = result.groupby("generating_state")["n"].transform("sum")
    result["fraction_within_generating_state"] = result["n"] / result["generating_state_n"]
    result.insert(0, "simulation", int(simulation))
    return result


def _retention_summary(retention: pd.DataFrame) -> pd.DataFrame:
    retention = retention.copy()
    retention["motion_coherence"] = pd.to_numeric(
        retention["motion_coherence"], errors="raise"
    ).round(12)
    retention["prior_std"] = pd.to_numeric(retention["prior_std"], errors="raise").round(12)
    totals = retention.groupby(
        ["source", "simulation", "motion_coherence", "prior_std"], as_index=False
    ).agg(total_n=("n", "sum"))
    retained_counts = retention[retention["retained"]].groupby(
        ["source", "simulation", "motion_coherence", "prior_std"], as_index=False
    )["n"].sum().rename(columns={"n": "retained_n_value"})
    totals = totals.merge(
        retained_counts,
        on=["source", "simulation", "motion_coherence", "prior_std"],
        how="left",
    ).rename(columns={"retained_n_value": "retained_n"})
    totals["retained_n"] = totals["retained_n"].fillna(0).astype(int)
    totals["retained_rate"] = totals["retained_n"] / totals["total_n"]
    observed = totals[totals["source"] == "observed"].set_index(["motion_coherence", "prior_std"])
    rows = []
    for keys, group in totals[totals["source"] == "simulated"].groupby(
        ["motion_coherence", "prior_std"], sort=True
    ):
        obs = observed.loc[keys, :]
        if isinstance(obs, pd.DataFrame):
            obs = obs.iloc[0]
        rates = group["retained_rate"].to_numpy(dtype=float)
        counts = group["retained_n"].to_numpy(dtype=float)
        rows.append({
            "motion_coherence": float(keys[0]),
            "prior_std": float(keys[1]),
            "condition": f"{float(keys[0]):g}/{float(keys[1]):g}",
            "observed_total_n": int(obs["total_n"]),
            "observed_retained_n": int(obs["retained_n"]),
            "observed_retained_rate": float(obs["retained_rate"]),
            "simulated_retained_n_mean": float(counts.mean()),
            "simulated_retained_rate_mean": float(rates.mean()),
            "simulated_retained_rate_ci_low": float(np.quantile(rates, 0.025)),
            "simulated_retained_rate_ci_high": float(np.quantile(rates, 0.975)),
        })
    return pd.DataFrame(rows)


def _coverage_table(metric_intervals: pd.DataFrame, model_label: str) -> pd.DataFrame:
    frame = metric_intervals.copy()
    frame["covered"] = frame["observed"].between(frame["simulated_ci_low"], frame["simulated_ci_high"])
    result = frame.groupby("metric", as_index=False).agg(cells=("covered", "size"), covered=("covered", "sum"))
    result["coverage_rate"] = result["covered"] / result["cells"]
    result.insert(0, "model", model_label)
    return result


def _sp_model_paths(out_dir: Path, model: str) -> dict[str, Path]:
    prefix = str(MODEL_SPECS[model]["prefix"])
    return {
        "summary": out_dir / f"{prefix}posterior_predictive_sp_condition_summary.csv",
        "histograms": out_dir / f"{prefix}posterior_predictive_sp_histograms.csv",
        "metric_intervals": out_dir / f"{prefix}posterior_predictive_sp_metric_intervals.csv",
        "histogram_intervals": out_dir / f"{prefix}posterior_predictive_sp_histogram_intervals.csv",
        "coverage": out_dir / f"{prefix}posterior_predictive_sp_coverage.csv",
        "retention": out_dir / f"{prefix}posterior_predictive_sp_retention.csv",
        "retention_summary": out_dir / f"{prefix}posterior_predictive_sp_retention_summary.csv",
        "classification": out_dir / f"{prefix}posterior_predictive_sp_classification_confusion.csv",
    }


def _manifest_files_valid(manifest: dict[str, Any], out_dir: Path) -> bool:
    if manifest.get("status") != "complete":
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


def _source_ppc_manifest(
    out_dir: Path,
    models: list[str],
    n_simulations: int,
    seed: int,
) -> dict[str, Any]:
    path = out_dir / "posterior_predictive_manifest.json"
    if not path.exists():
        raise RuntimeError("The completed all-trial PPC manifest is required")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("The all-trial PPC stage is not complete")
    if manifest.get("models") != models:
        raise RuntimeError("Requested S/P models do not match the completed all-trial PPC")
    if int(manifest.get("n_simulations", -1)) != int(n_simulations) or int(manifest.get("seed", -1)) != int(seed):
        raise RuntimeError("S/P settings do not match the completed all-trial PPC")
    if not _manifest_files_valid(manifest, out_dir):
        raise RuntimeError("An all-trial PPC output is missing or does not match its recorded hash")
    return manifest


def _observed_posterior_path(out_dir: Path, model: str) -> Path:
    return out_dir / ("posterior_states.csv" if model == "static_hmm" else "covariate_hmm_posterior_states.csv")


def _run_sp_model(
    data: DataBundle,
    fit: Any,
    model: str,
    out_dir: Path,
    work_dir: Path,
    n_simulations: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label = str(MODEL_SPECS[model]["label"])
    covariate = model == "covariate_hmm"
    observed_decoded, observed_validation = decode_smoothed_marginal_states(
        data,
        fit.params,
        data.df["estimate_deg"].to_numpy(dtype=float),
        model=model,
    )
    existing = pd.read_csv(_observed_posterior_path(out_dir, model)).sort_values("row_index")
    if not np.array_equal(existing["row_index"].to_numpy(dtype=int), np.arange(len(data.df))):
        raise RuntimeError(f"{label} observed posterior rows are incomplete")
    if not np.array_equal(existing["state_map"].to_numpy(), observed_decoded["state_map"].to_numpy()):
        raise RuntimeError(f"{label} observed marginal-MAP states do not match checkpoint decoding")
    if "state_map_method" in existing and set(existing["state_map_method"].dropna()) != {SP_METHOD}:
        raise RuntimeError(f"{label} observed posterior method is not marginal-MAP smoothing")

    observed_keep = observed_decoded["state_map"].to_numpy() != "L_lapse"
    observed_summary, observed_hist = _conditional_summaries(
        data,
        data.df["estimate_deg"].to_numpy(dtype=float),
        observed_keep,
        source="observed",
        simulation=-1,
    )
    observed_retention = _retention_table(
        data,
        observed_decoded["state_map"].to_numpy(),
        source="observed",
        simulation=-1,
    )

    model_work = work_dir / model
    model_work.mkdir(parents=True, exist_ok=True)
    summary_parts: list[pd.DataFrame] = []
    histogram_parts: list[pd.DataFrame] = []
    retention_parts: list[pd.DataFrame] = []
    classification_parts: list[pd.DataFrame] = []
    validation_draws: list[dict[str, float]] = []
    raw_path = _model_paths(out_dir, model)["responses"]
    columns = ["simulation", "row_index", "estimate_deg", "state", "transition_prev_error_deg"]
    reader = pd.read_csv(raw_path, usecols=columns, chunksize=len(data.df))
    seen = 0
    for expected_simulation, draw in enumerate(reader):
        simulation_values = draw["simulation"].unique()
        if len(draw) != len(data.df) or list(simulation_values) != [expected_simulation]:
            raise RuntimeError(f"{label} raw simulation chunk {expected_simulation} is malformed")
        if not np.array_equal(draw["row_index"].to_numpy(dtype=int), np.arange(len(data.df))):
            raise RuntimeError(f"{label} simulation {expected_simulation} trial order is invalid")
        summary_path = model_work / f"draw_{expected_simulation:04d}_summary.csv"
        histogram_path = model_work / f"draw_{expected_simulation:04d}_histograms.csv"
        retention_path = model_work / f"draw_{expected_simulation:04d}_retention.csv"
        classification_path = model_work / f"draw_{expected_simulation:04d}_classification.csv"
        validation_path = model_work / f"draw_{expected_simulation:04d}_validation.json"
        sidecars = [summary_path, histogram_path, retention_path, classification_path, validation_path]
        ready = resume and all(path.exists() and path.stat().st_size > 0 for path in sidecars)
        if ready:
            summary = pd.read_csv(summary_path)
            histograms = pd.read_csv(histogram_path)
            retention = pd.read_csv(retention_path)
            classification = pd.read_csv(classification_path)
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        else:
            response_deg = draw["estimate_deg"].to_numpy(dtype=float)
            decoded, validation = decode_smoothed_marginal_states(
                data,
                fit.params,
                response_deg,
                model=model,
                recorded_prev_error_deg=(
                    draw["transition_prev_error_deg"].to_numpy(dtype=float) if covariate else None
                ),
            )
            decoded_state = decoded["state_map"].to_numpy()
            keep = decoded_state != "L_lapse"
            summary, histograms = _conditional_summaries(
                data,
                response_deg,
                keep,
                source="simulated",
                simulation=expected_simulation,
            )
            retention = _retention_table(
                data,
                decoded_state,
                source="simulated",
                simulation=expected_simulation,
            )
            classification = _confusion_table(
                draw["state"].to_numpy(),
                decoded_state,
                expected_simulation,
            )
            _atomic_csv(summary, summary_path)
            _atomic_csv(histograms, histogram_path)
            _atomic_csv(retention, retention_path)
            _atomic_csv(classification, classification_path)
            atomic_write_json(validation, validation_path)
        summary_parts.append(summary)
        histogram_parts.append(histograms)
        retention_parts.append(retention)
        classification_parts.append(classification)
        validation_draws.append(validation)
        seen += 1
        print(f"S/P PPC {model}: decoded draw {seen}/{n_simulations}", flush=True)
    if seen != n_simulations:
        raise RuntimeError(f"{label} contains {seen} simulations, expected {n_simulations}")

    summary = pd.concat([observed_summary, *summary_parts], ignore_index=True)
    histograms = pd.concat([observed_hist, *histogram_parts], ignore_index=True)
    retention = pd.concat([observed_retention, *retention_parts], ignore_index=True)
    classification = pd.concat(classification_parts, ignore_index=True)
    metric_intervals = ppc_metric_intervals(summary)
    histogram_intervals = ppc_histogram_intervals(histograms)
    metric_intervals["n_simulations"] = n_simulations
    histogram_intervals["n_simulations"] = n_simulations
    retention_summary = _retention_summary(retention)
    retention_summary["n_simulations"] = n_simulations
    coverage = _coverage_table(metric_intervals, label)
    for frame in [summary, histograms, retention, classification, metric_intervals, histogram_intervals, retention_summary]:
        frame.insert(0, "model", label)

    frames = {
        "summary": summary,
        "histograms": histograms,
        "metric_intervals": metric_intervals,
        "histogram_intervals": histogram_intervals,
        "coverage": coverage,
        "retention": retention,
        "retention_summary": retention_summary,
        "classification": classification,
    }
    records = []
    paths = _sp_model_paths(out_dir, model)
    for key, frame in frames.items():
        path = _atomic_csv(frame, paths[key])
        records.append(_file_record(path, out_dir, len(frame), len(frame.columns)))
    validation = {
        "draws": n_simulations,
        "rows_per_draw": len(data.df),
        "sequences_per_draw": len(data.sequences),
        "method": SP_METHOD,
        "observed_retained_trials": int(observed_keep.sum()),
        "observed_excluded_lapse_trials": int((~observed_keep).sum()),
        "max_posterior_row_sum_error": max(
            [observed_validation["max_posterior_row_sum_error"]]
            + [float(row["max_posterior_row_sum_error"]) for row in validation_draws]
        ),
        "max_simulated_previous_error_error_rad": max(
            float(row["max_simulated_previous_error_error_rad"]) for row in validation_draws
        ),
        "minimum_retained_condition_count": int(summary["n"].min()),
    }
    return records, validation


def run_lapse_excluded_posterior_predictive_checks(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    models: Iterable[str] = ("static_hmm", "covariate_hmm"),
    n_simulations: int = 100,
    seed: int = 42,
    resume: bool = True,
    update_parent_manifest: bool = True,
) -> dict[str, Any]:
    """Run the symmetric S/P-only sensitivity check from existing PPC draws."""
    csv_path = Path(csv_path).resolve()
    out_dir = Path(out_dir).resolve()
    model_list = list(dict.fromkeys(models))
    unknown = [model for model in model_list if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown S/P PPC models: {unknown}")
    source_manifest = _source_ppc_manifest(out_dir, model_list, n_simulations, seed)
    parent_path = out_dir / "run_manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if sha256_file(csv_path) != parent.get("data_sha256") or parent.get("run_key") != source_manifest.get("parent_run_key"):
        raise RuntimeError("Dataset or run key does not match the completed PPC")
    data = load_direction_data(csv_path)
    fits: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    raw_hashes: dict[str, str] = {}
    for model in model_list:
        checkpoint = out_dir / "checkpoints" / str(MODEL_SPECS[model]["checkpoint"])
        fit = load_fit_checkpoint(checkpoint, str(parent["run_key"]))
        if fit is None:
            raise RuntimeError(f"Missing or incompatible checkpoint: {checkpoint}")
        fits[model] = fit
        checkpoint_hashes[model] = sha256_file(checkpoint)
        raw_hashes[model] = sha256_file(_model_paths(out_dir, model)["responses"])
    sp_key = stable_hash({
        "schema_version": SP_SCHEMA_VERSION,
        "parent_run_key": parent["run_key"],
        "source_ppc_key": source_manifest["ppc_key"],
        "models": model_list,
        "n_simulations": int(n_simulations),
        "seed": int(seed),
        "method": SP_METHOD,
        "checkpoint_hashes": checkpoint_hashes,
        "raw_hashes": raw_hashes,
    })
    manifest_path = out_dir / "posterior_predictive_sp_manifest.json"
    if resume and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("sp_key") == sp_key and _manifest_files_valid(existing_manifest, out_dir):
            if update_parent_manifest:
                parent.setdefault("stages", {})["posterior_predictive_sp_sensitivity"] = {
                    "status": "complete",
                    "updated_at": utc_now(),
                    "models": model_list,
                    "simulations_per_model": int(n_simulations),
                    "seed": int(seed),
                    "method": SP_METHOD,
                    "manifest": str(manifest_path.resolve()),
                }
                atomic_write_json(parent, parent_path)
            print("S/P PPC: reusing validated completed outputs", flush=True)
            return existing_manifest

    work_dir = out_dir / ".ppc_sp_work" / sp_key
    work_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for model in model_list:
        records, model_validation = _run_sp_model(
            data,
            fits[model],
            model,
            out_dir,
            work_dir,
            int(n_simulations),
            resume,
        )
        files.extend(records)
        validation[model] = model_validation

    combined_specs = {
        "posterior_predictive_sp_model_metric_intervals.csv": "metric_intervals",
        "posterior_predictive_sp_model_histogram_intervals.csv": "histogram_intervals",
        "posterior_predictive_sp_model_coverage.csv": "coverage",
        "posterior_predictive_sp_model_retention.csv": "retention",
        "posterior_predictive_sp_model_retention_summary.csv": "retention_summary",
        "posterior_predictive_sp_model_classification_confusion.csv": "classification",
    }
    for filename, key in combined_specs.items():
        frame = pd.concat(
            [pd.read_csv(_sp_model_paths(out_dir, model)[key]) for model in model_list],
            ignore_index=True,
        )
        path = _atomic_csv(frame, out_dir / filename)
        files.append(_file_record(path, out_dir, len(frame), len(frame.columns)))

    checkpoint_after = {
        model: sha256_file(out_dir / "checkpoints" / str(MODEL_SPECS[model]["checkpoint"]))
        for model in model_list
    }
    raw_after = {model: sha256_file(_model_paths(out_dir, model)["responses"]) for model in model_list}
    if checkpoint_after != checkpoint_hashes or raw_after != raw_hashes:
        raise RuntimeError("A checkpoint or raw all-trial PPC file changed during S/P analysis")
    manifest = {
        "schema_version": SP_SCHEMA_VERSION,
        "status": "complete",
        "generated_at": utc_now(),
        "fit_independent": True,
        "response_simulation_reused": True,
        "sp_key": sp_key,
        "parent_run_key": parent["run_key"],
        "source_ppc_key": source_manifest["ppc_key"],
        "models": model_list,
        "n_simulations": int(n_simulations),
        "seed": int(seed),
        "exclusion_rule": "decoded_state_map_equals_L_lapse",
        "state_path_summary": SP_METHOD,
        "run_length_ppc_modified": False,
        "checkpoint_hashes_before": checkpoint_hashes,
        "checkpoint_hashes_after": checkpoint_after,
        "raw_ppc_hashes_before": raw_hashes,
        "raw_ppc_hashes_after": raw_after,
        "validation": validation,
        "files": sorted(files, key=lambda row: row["path"]),
    }
    atomic_write_json(manifest, manifest_path)
    if update_parent_manifest:
        parent.setdefault("stages", {})["posterior_predictive_sp_sensitivity"] = {
            "status": "complete",
            "updated_at": utc_now(),
            "models": model_list,
            "simulations_per_model": int(n_simulations),
            "seed": int(seed),
            "method": SP_METHOD,
            "manifest": str(manifest_path.resolve()),
        }
        atomic_write_json(parent, parent_path)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    return manifest
