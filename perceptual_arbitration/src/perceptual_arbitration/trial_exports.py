from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy.special import i0e, i1e, logsumexp

from .checkpoints import load_fit_checkpoint
from .circular import LOG_TWO_PI, circ_diff, log_i0, rad2deg, wrap_rad
from .covariate_hmm import (
    CovariateHMMParams,
    forward_backward_timevarying,
    transition_logA_sequence,
)
from .data import (
    DataBundle,
    load_direction_data,
    transition_scaler_from_sequences,
    with_transition_scaler,
)
from .hmm import HMMParams, K, forward_backward_scaled
from .model_selection import make_cv_splits
from .run_metadata import atomic_write_json, sha256_file, utc_now
from .serial_dependence import previous_arrays


SCHEMA_VERSION = 1
STATE_KEYS = ("sensory", "prior", "lapse")
STATE_LABELS = ("S_sensory", "P_prior", "L_lapse")
TRANSITION_COLUMNS = tuple(
    f"transition_p_{source}_to_{target}"
    for source in STATE_KEYS
    for target in STATE_KEYS
)
XI_COLUMNS = tuple(
    f"smoothed_transition_posterior_{source}_to_{target}"
    for source in STATE_KEYS
    for target in STATE_KEYS
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    family: Literal["independent", "serial", "hmm", "covariate_hmm"]
    final_checkpoint: str
    cv_checkpoint: str


GLOBAL_MODELS = (
    ModelSpec(
        "independent_switching",
        "Independent_switching",
        "independent",
        "final_independent_switching",
        "cv_fold_{fold}_independent_switching",
    ),
    ModelSpec(
        "serial_stimulus",
        "Serial_stim_independent_switching",
        "serial",
        "final_serial_stim",
        "cv_fold_{fold}_serial_stim",
    ),
    ModelSpec(
        "serial_response",
        "Serial_resp_independent_switching",
        "serial",
        "final_serial_resp",
        "cv_fold_{fold}_serial_resp",
    ),
    ModelSpec(
        "serial_both",
        "Serial_both_independent_switching",
        "serial",
        "final_serial_both",
        "cv_fold_{fold}_serial_both",
    ),
    ModelSpec(
        "static_hmm",
        "HMM_static",
        "hmm",
        "final_hmm_static",
        "cv_fold_{fold}_hmm_static",
    ),
    ModelSpec(
        "covariate_hmm",
        "Covariate_HMM",
        "covariate_hmm",
        "final_covariate_hmm",
        "cv_fold_{fold}_covariate_hmm",
    ),
)


def _signed_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.rad2deg(circ_diff(a, b))


def _selected_indices(data: DataBundle, seq_ids: Iterable[int]) -> np.ndarray:
    ids = [int(value) for value in seq_ids]
    if not ids:
        return np.array([], dtype=int)
    return np.concatenate([data.sequences[sid] for sid in ids]).astype(int)


def build_trial_base(data: DataBundle) -> pd.DataFrame:
    """Create the model-independent, audit-ready trial table."""
    source = data.df
    n = len(source)
    seq_id = np.full(n, -1, dtype=int)
    sequence_position = np.full(n, -1, dtype=int)
    sequence_length = np.zeros(n, dtype=int)
    is_start = np.zeros(n, dtype=bool)
    for sid, idx in enumerate(data.sequences):
        seq_id[idx] = sid
        sequence_position[idx] = np.arange(len(idx), dtype=int)
        sequence_length[idx] = len(idx)
        is_start[idx[0]] = True

    def values(name: str, default: Any = np.nan) -> np.ndarray:
        if name in source:
            return source[name].to_numpy(copy=True)
        return np.full(n, default)

    signed_error = _signed_degrees(data.y, data.theta)
    prior_error = _signed_degrees(data.y, data.prior_mu)
    conflict_signed = _signed_degrees(data.theta, data.prior_mu)
    base = pd.DataFrame({
        "raw_row_index": values("raw_row_index").astype(int),
        "analysis_row_index": np.arange(n, dtype=int),
        "seq_id": seq_id,
        "subject_id": values("subject_id"),
        "session_id": values("session_id"),
        "run_id": values("run_id"),
        "block_id": values("run_id"),
        "trial_index": values("trial_index"),
        "sequence_trial_index": sequence_position,
        "sequence_length": sequence_length,
        "trial_time": values("trial_time"),
        "experiment_name": values("experiment_name", ""),
        "experiment_id": values("experiment_id"),
        "is_sequence_start": is_start,
        "motion_direction_deg": values("motion_direction"),
        "motion_coherence": values("motion_coherence"),
        "prior_mean_deg": values("prior_mean"),
        "prior_std_deg": values("prior_std"),
        "response_arrow_start_angle_deg": values("response_arrow_start_angle"),
        "estimate_x": values("estimate_x"),
        "estimate_y": values("estimate_y"),
        "estimated_direction_deg": values("estimate_deg"),
        "reaction_time": values("reaction_time"),
        "raw_response_time": values("raw_response_time"),
        "signed_stimulus_error_deg": signed_error,
        "absolute_stimulus_error_deg": np.abs(signed_error),
        "response_minus_prior_deg": prior_error,
        "absolute_prior_error_deg": np.abs(prior_error),
        "signed_stimulus_prior_conflict_deg": conflict_signed,
        "stimulus_prior_conflict_deg": np.abs(conflict_signed),
    })

    previous_columns = {
        "previous_motion_direction_deg": base["motion_direction_deg"].to_numpy(dtype=float),
        "previous_estimated_direction_deg": base["estimated_direction_deg"].to_numpy(dtype=float),
        "previous_signed_stimulus_error_deg": signed_error,
        "previous_absolute_stimulus_error_deg": np.abs(signed_error),
        "previous_stimulus_prior_conflict_deg": np.abs(conflict_signed),
        "previous_coherence": base["motion_coherence"].to_numpy(dtype=float),
    }
    for name, current in previous_columns.items():
        previous = np.full(n, np.nan, dtype=float)
        for idx in data.sequences:
            previous[idx[1:]] = current[idx[:-1]]
        base[name] = previous
    return base


def _emission_log_densities(
    data: DataBundle,
    params: Any,
    idx: np.ndarray,
    sensory_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensory_kappa = np.asarray(params.kappa_s, dtype=float)[data.coh_idx[idx]]
    prior_kappa = np.asarray(params.kappa_p, dtype=float)[data.prior_idx[idx]]
    sensory_delta = circ_diff(data.y[idx], sensory_centers[idx])
    log_b = np.empty((len(idx), K), dtype=float)
    log_b[:, 0] = sensory_kappa * np.cos(sensory_delta) - LOG_TWO_PI - log_i0(sensory_kappa)
    log_b[:, 1] = prior_kappa * data.cos_p[idx] - LOG_TWO_PI - log_i0(prior_kappa)
    log_b[:, 2] = -LOG_TWO_PI
    return log_b, sensory_kappa, prior_kappa


def _predictive_moments(
    state_probabilities: np.ndarray,
    sensory_center: np.ndarray,
    prior_center: np.ndarray,
    sensory_kappa: np.ndarray,
    prior_kappa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensory_resultant = i1e(sensory_kappa) / np.maximum(i0e(sensory_kappa), 1e-300)
    prior_resultant = i1e(prior_kappa) / np.maximum(i0e(prior_kappa), 1e-300)
    vector = (
        state_probabilities[:, 0] * sensory_resultant * np.exp(1j * sensory_center)
        + state_probabilities[:, 1] * prior_resultant * np.exp(1j * prior_center)
    )
    resultant = np.clip(np.abs(vector), 0.0, 1.0)
    mean = rad2deg(np.angle(vector))
    mean[resultant < 1e-15] = np.nan
    circular_sd = np.rad2deg(np.sqrt(np.maximum(-2.0 * np.log(np.maximum(resultant, 1e-300)), 0.0)))
    return mean, resultant, circular_sd


def _common_score_frame(
    data: DataBundle,
    idx: np.ndarray,
    log_b: np.ndarray,
    sensory_kappa: np.ndarray,
    prior_kappa: np.ndarray,
    sensory_centers: np.ndarray,
    predictive: np.ndarray,
    filtered: np.ndarray,
    trial_log_density: np.ndarray,
    sequence_cumulative: np.ndarray,
    sequence_total: np.ndarray,
) -> pd.DataFrame:
    mean, resultant, circular_sd = _predictive_moments(
        predictive,
        sensory_centers[idx],
        data.prior_mu[idx],
        sensory_kappa,
        prior_kappa,
    )
    log_contribution = np.log(np.maximum(predictive, 1e-300)) + log_b
    frame = pd.DataFrame(index=idx)
    frame.index.name = "analysis_row_index"
    frame["sensory_center_deg"] = rad2deg(sensory_centers[idx])
    frame["prior_center_deg"] = rad2deg(data.prior_mu[idx])
    frame["sensory_kappa"] = sensory_kappa
    frame["prior_kappa"] = prior_kappa
    for state_index, state in enumerate(STATE_KEYS):
        frame[f"emission_log_density_{state}"] = log_b[:, state_index]
        frame[f"emission_density_{state}"] = np.exp(log_b[:, state_index])
        frame[f"predictive_p_{state}"] = predictive[:, state_index]
        frame[f"state_log_contribution_{state}"] = log_contribution[:, state_index]
        frame[f"filtered_p_{state}"] = filtered[:, state_index]
    frame["state_map_filtered"] = np.asarray(STATE_LABELS, dtype=object)[np.argmax(filtered, axis=1)]
    frame["trial_predictive_density"] = np.exp(trial_log_density)
    frame["trial_log_predictive_density"] = trial_log_density
    frame["sequence_cumulative_log_likelihood"] = sequence_cumulative
    frame["sequence_log_likelihood"] = sequence_total
    frame["predictive_circular_mean_deg"] = mean
    frame["predictive_resultant_length"] = resultant
    frame["predictive_circular_sd_deg"] = circular_sd
    return frame


def score_independent_or_serial(
    data: DataBundle,
    params: Any,
    seq_ids: Iterable[int],
    *,
    serial: bool,
) -> pd.DataFrame:
    """Score an independent mixture or one-back serial mixture trial by trial."""
    seq_ids = [int(value) for value in seq_ids]
    idx = _selected_indices(data, seq_ids)
    sensory_centers = data.theta.copy()
    alpha_stim = float(getattr(params, "alpha_stim", 0.0))
    alpha_resp = float(getattr(params, "alpha_resp", 0.0))
    serial_shift = np.zeros(len(data.df), dtype=float)
    if serial:
        previous_theta, previous_y = previous_arrays(data)
        serial_shift = (
            alpha_stim * circ_diff(previous_theta, data.theta)
            + alpha_resp * circ_diff(previous_y, data.theta)
        )
        sensory_centers = wrap_rad(data.theta + serial_shift)

    log_b, sensory_kappa, prior_kappa = _emission_log_densities(
        data, params, idx, sensory_centers
    )
    weights = np.asarray(getattr(params, "weights", getattr(params, "w", None)), dtype=float)
    predictive = np.broadcast_to(weights, (len(idx), K)).copy()
    log_joint = np.log(np.maximum(predictive, 1e-300)) + log_b
    trial_log_density = logsumexp(log_joint, axis=1)
    filtered = np.exp(log_joint - trial_log_density[:, None])
    cumulative = np.empty(len(idx), dtype=float)
    total = np.empty(len(idx), dtype=float)
    cursor = 0
    for sid in seq_ids:
        length = len(data.sequences[sid])
        part = trial_log_density[cursor:cursor + length]
        cumulative[cursor:cursor + length] = np.cumsum(part)
        total[cursor:cursor + length] = part.sum()
        cursor += length

    frame = _common_score_frame(
        data,
        idx,
        log_b,
        sensory_kappa,
        prior_kappa,
        sensory_centers,
        predictive,
        filtered,
        trial_log_density,
        cumulative,
        total,
    )
    for state_index, state in enumerate(STATE_KEYS):
        frame[f"mixture_weight_{state}"] = weights[state_index]
    if serial:
        start_mask = np.zeros(len(data.df), dtype=bool)
        for sequence in data.sequences:
            start_mask[sequence[0]] = True
        frame["alpha_previous_stimulus"] = alpha_stim
        frame["alpha_previous_response"] = alpha_resp
        frame["serial_sensory_center_shift_deg"] = np.rad2deg(serial_shift[idx])
        frame["unshifted_sensory_center_deg"] = rad2deg(data.theta[idx])
        frame["shifted_sensory_center_deg"] = rad2deg(sensory_centers[idx])
        frame["serial_history_available"] = ~start_mask[idx]
        frame["serial_sequence_start_current_trial_fallback"] = start_mask[idx]
    return frame


def score_hmm(
    data: DataBundle,
    params: HMMParams | CovariateHMMParams,
    seq_ids: Iterable[int],
    *,
    covariate: bool,
) -> pd.DataFrame:
    """Export online filtering and whole-sequence smoothing for an HMM."""
    seq_ids = [int(value) for value in seq_ids]
    idx_all = _selected_indices(data, seq_ids)
    sensory_centers = data.theta.copy()
    log_b_all, sensory_kappa, prior_kappa = _emission_log_densities(
        data, params, idx_all, sensory_centers
    )
    position = {int(row): pos for pos, row in enumerate(idx_all)}
    predictive = np.empty((len(idx_all), K), dtype=float)
    filtered = np.empty((len(idx_all), K), dtype=float)
    smoothed = np.empty((len(idx_all), K), dtype=float)
    trial_log_density = np.empty(len(idx_all), dtype=float)
    cumulative = np.empty(len(idx_all), dtype=float)
    total = np.empty(len(idx_all), dtype=float)
    transitions = np.full((len(idx_all), K, K), np.nan, dtype=float)
    xi_rows = np.full((len(idx_all), K, K), np.nan, dtype=float)

    for sid in seq_ids:
        seq_idx = data.sequences[sid]
        locations = np.array([position[int(row)] for row in seq_idx], dtype=int)
        log_b = log_b_all[locations]
        if covariate:
            assert isinstance(params, CovariateHMMParams)
            log_a_sequence = transition_logA_sequence(params, data.X_transition[seq_idx])
            sequence_ll, gamma, xi = forward_backward_timevarying(
                log_b,
                np.log(np.maximum(params.pi, 1e-300)),
                log_a_sequence,
            )
            a_sequence = np.exp(log_a_sequence)
        else:
            assert isinstance(params, HMMParams)
            sequence_ll, gamma, xi = forward_backward_scaled(log_b, params.pi, params.A)
            a_sequence = np.broadcast_to(params.A, (max(len(seq_idx) - 1, 0), K, K))

        sequence_predictive = np.empty((len(seq_idx), K), dtype=float)
        sequence_filtered = np.empty((len(seq_idx), K), dtype=float)
        increments = np.empty(len(seq_idx), dtype=float)
        sequence_predictive[0] = params.pi
        for trial in range(len(seq_idx)):
            log_joint = np.log(np.maximum(sequence_predictive[trial], 1e-300)) + log_b[trial]
            increments[trial] = logsumexp(log_joint)
            sequence_filtered[trial] = np.exp(log_joint - increments[trial])
            if trial + 1 < len(seq_idx):
                sequence_predictive[trial + 1] = sequence_filtered[trial] @ a_sequence[trial]

        if not np.isclose(increments.sum(), sequence_ll, rtol=0.0, atol=1e-8):
            raise RuntimeError(
                f"HMM filtering increments do not reproduce sequence {sid} likelihood: "
                f"{increments.sum()} versus {sequence_ll}"
            )
        predictive[locations] = sequence_predictive
        filtered[locations] = sequence_filtered
        smoothed[locations] = gamma
        trial_log_density[locations] = increments
        cumulative[locations] = np.cumsum(increments)
        total[locations] = increments.sum()
        if len(seq_idx) > 1:
            transitions[locations[1:]] = a_sequence
            xi_rows[locations[1:]] = xi

    frame = _common_score_frame(
        data,
        idx_all,
        log_b_all,
        sensory_kappa,
        prior_kappa,
        sensory_centers,
        predictive,
        filtered,
        trial_log_density,
        cumulative,
        total,
    )
    for state_index, state in enumerate(STATE_KEYS):
        frame[f"initial_p_{state}"] = float(params.pi[state_index])
        frame[f"smoothed_p_{state}"] = smoothed[:, state_index]
    frame["state_map_smoothed"] = np.asarray(STATE_LABELS, dtype=object)[np.argmax(smoothed, axis=1)]
    frame["smoothed_probabilities_use_complete_sequence"] = True
    frame["transition_applies"] = ~np.isnan(transitions[:, 0, 0])
    for column_index, column in enumerate(TRANSITION_COLUMNS):
        source, target = divmod(column_index, K)
        frame[column] = transitions[:, source, target]
    for column_index, column in enumerate(XI_COLUMNS):
        source, target = divmod(column_index, K)
        frame[column] = xi_rows[:, source, target]

    if covariate:
        raw_names = data.transition_names[1:]
        for column_index, name in enumerate(raw_names):
            frame[f"transition_covariate_{name}_raw"] = data.X_transition_raw[idx_all, column_index]
            frame[f"transition_covariate_{name}_z"] = data.X_transition[idx_all, column_index + 1]
            frame[f"transition_scaler_mean_{name}"] = data.transition_means[column_index]
            frame[f"transition_scaler_sd_{name}"] = data.transition_sds[column_index]
    return frame


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> Path:
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


def _load_required_fit(out_dir: Path, checkpoint_name: str, run_key: str) -> tuple[Any, Path]:
    path = out_dir / "checkpoints" / f"{checkpoint_name}.joblib"
    fit = load_fit_checkpoint(path, run_key)
    if fit is None:
        raise RuntimeError(
            f"Missing or incompatible checkpoint {path}; no model will be refitted by the exporter"
        )
    return fit, path


def _score_model(
    spec: ModelSpec,
    data: DataBundle,
    fit: Any,
    seq_ids: Iterable[int],
) -> pd.DataFrame:
    if spec.family == "independent":
        return score_independent_or_serial(data, fit.params, seq_ids, serial=False)
    if spec.family == "serial":
        return score_independent_or_serial(data, fit.params, seq_ids, serial=True)
    if spec.family == "hmm":
        return score_hmm(data, fit.params, seq_ids, covariate=False)
    if spec.family == "covariate_hmm":
        return score_hmm(data, fit.params, seq_ids, covariate=True)
    raise ValueError(f"Unsupported model family: {spec.family}")


def _make_detailed_frame(
    base: pd.DataFrame,
    score: pd.DataFrame,
    *,
    spec: ModelSpec,
    scope: str,
    fold: int | None,
    fit: Any,
    checkpoint_path: Path,
    out_dir: Path,
    run_key: str,
    parameter_source: str,
    scaler_provenance: str,
    training_sequence_count: int,
    training_trial_count: int,
    fit_subject_id: Any = np.nan,
) -> pd.DataFrame:
    score = score.sort_index()
    idx = score.index.to_numpy(dtype=int)
    trial = base.set_index("analysis_row_index", drop=False).loc[idx].reset_index(drop=True)
    provenance = pd.DataFrame({
        "model_key": spec.key,
        "model": spec.model_name,
        "model_family": spec.family,
        "fit_scope": scope,
        "fold": pd.array([fold] * len(score), dtype="Int64"),
        "relative_checkpoint_path": checkpoint_path.relative_to(out_dir).as_posix(),
        "run_key": run_key,
        "fit_seed": int(fit.seed),
        "fit_converged": bool(fit.converged),
        "fit_n_iter": int(fit.n_iter),
        "fit_train_log_likelihood_recorded": float(fit.train_loglik),
        "parameter_source": parameter_source,
        "training_sequence_count": int(training_sequence_count),
        "training_trial_count": int(training_trial_count),
        "transition_scaler_provenance": scaler_provenance,
        "fit_subject_id": fit_subject_id,
        "trial_score_role": (
            "held_out_out_of_fold_prediction"
            if scope == "oof"
            else "descriptive_in_sample_reconstruction"
        ),
    })
    return pd.concat([trial, provenance, score.reset_index(drop=True)], axis=1)


def _parameter_rows(
    spec: ModelSpec,
    fit: Any,
    data: DataBundle,
    *,
    scope: str,
    fold: int | None,
    subject_id: Any,
    checkpoint_path: Path,
    out_dir: Path,
    scaler_provenance: str,
) -> list[dict[str, Any]]:
    params = fit.params
    common = {
        "model_key": spec.key,
        "model": spec.model_name,
        "fit_scope": scope,
        "fold": fold,
        "subject_id": subject_id,
        "relative_checkpoint_path": checkpoint_path.relative_to(out_dir).as_posix(),
        "fit_seed": int(fit.seed),
        "transition_scaler_provenance": scaler_provenance,
    }
    rows: list[dict[str, Any]] = []

    def add(group: str, parameter: str, value: float, level_1="", level_2="", covariate=""):
        rows.append({
            **common,
            "parameter_group": group,
            "parameter": parameter,
            "level_1": level_1,
            "level_2": level_2,
            "covariate": covariate,
            "value": float(value),
        })

    if hasattr(params, "w") or hasattr(params, "weights"):
        weights = np.asarray(getattr(params, "weights", getattr(params, "w", None)), dtype=float)
        for state, value in zip(STATE_LABELS, weights):
            add("component_weights", "weight", value, level_1=state)
    if hasattr(params, "pi"):
        for state, value in zip(STATE_LABELS, np.asarray(params.pi, dtype=float)):
            add("initial_probabilities", "pi", value, level_1=state)
    if hasattr(params, "A"):
        for source, row in zip(STATE_LABELS, np.asarray(params.A, dtype=float)):
            for target, value in zip(STATE_LABELS, row):
                add("transition_matrix", "A", value, level_1=source, level_2=target)
    for coherence, value in zip(data.coh_values, np.asarray(params.kappa_s, dtype=float)):
        add("emission_concentrations", "sensory_kappa", value, level_1=f"coherence_{coherence:g}")
    for prior_width, value in zip(data.prior_values, np.asarray(params.kappa_p, dtype=float)):
        add("emission_concentrations", "prior_kappa", value, level_1=f"prior_std_{prior_width:g}")
    if hasattr(params, "alpha_stim"):
        add("serial_coefficients", "alpha_previous_stimulus", params.alpha_stim)
        add("serial_coefficients", "alpha_previous_response", params.alpha_resp)
    if hasattr(params, "B"):
        for i, source in enumerate(STATE_LABELS):
            for j, target in enumerate(STATE_LABELS):
                for p, covariate in enumerate(data.transition_names):
                    add(
                        "transition_softmax_coefficients",
                        "B",
                        params.B[i, j, p],
                        level_1=source,
                        level_2=target,
                        covariate=covariate,
                    )
        for p, name in enumerate(data.transition_names[1:]):
            add("transition_scaler", "mean", data.transition_means[p], covariate=name)
            add("transition_scaler", "sd", data.transition_sds[p], covariate=name)
    return rows


def _fit_metadata_row(
    spec: ModelSpec,
    fit: Any,
    *,
    scope: str,
    fold: int | None,
    subject_id: Any,
    checkpoint_path: Path,
    out_dir: Path,
    scored_sequence_count: int,
    scored_trial_count: int,
    training_sequence_count: int,
    training_trial_count: int,
    recomputed_scored_log_likelihood: float,
    scaler_provenance: str,
) -> dict[str, Any]:
    return {
        "model_key": spec.key,
        "model": spec.model_name,
        "model_family": spec.family,
        "fit_scope": scope,
        "fold": fold,
        "subject_id": subject_id,
        "relative_checkpoint_path": checkpoint_path.relative_to(out_dir).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "fit_seed": int(fit.seed),
        "fit_converged": bool(fit.converged),
        "fit_n_iter": int(fit.n_iter),
        "recorded_training_log_likelihood": float(fit.train_loglik),
        "recomputed_scored_log_likelihood": recomputed_scored_log_likelihood,
        "recorded_minus_recomputed_when_same_sample": (
            float(fit.train_loglik) - recomputed_scored_log_likelihood
            if scope in {"final", "subject_final"}
            else np.nan
        ),
        "training_sequence_count": training_sequence_count,
        "training_trial_count": training_trial_count,
        "scored_sequence_count": scored_sequence_count,
        "scored_trial_count": scored_trial_count,
        "transition_scaler_provenance": scaler_provenance,
        "restart_diagnostics_count": len(fit.restart_diagnostics or []),
    }


def _compact_scores(score: pd.DataFrame, model_key: str) -> pd.DataFrame:
    compact = score[[
        "trial_log_predictive_density",
        "trial_predictive_density",
        "filtered_p_sensory",
        "filtered_p_prior",
        "filtered_p_lapse",
        "state_map_filtered",
    ]].copy()
    compact.columns = [f"{model_key}_{column}" for column in compact.columns]
    compact.index.name = "analysis_row_index"
    return compact


def _comparison_frame(
    base: pd.DataFrame,
    compact_by_model: dict[str, pd.DataFrame],
    fold_by_row: pd.Series | None,
    scope: str,
) -> pd.DataFrame:
    comparison = base[[
        "raw_row_index",
        "analysis_row_index",
        "seq_id",
        "subject_id",
        "session_id",
        "run_id",
        "block_id",
        "trial_index",
        "is_sequence_start",
    ]].set_index("analysis_row_index", drop=False)
    comparison.insert(9, "fit_scope", scope)
    if fold_by_row is None:
        comparison.insert(10, "fold", pd.array([None] * len(comparison), dtype="Int64"))
    else:
        comparison.insert(
            10,
            "fold",
            pd.array(fold_by_row.reindex(comparison.index).to_numpy(), dtype="Int64"),
        )
    for spec in GLOBAL_MODELS:
        comparison = comparison.join(compact_by_model[spec.key], how="left")
    baseline = comparison["independent_switching_trial_log_predictive_density"]
    for spec in GLOBAL_MODELS:
        comparison[f"{spec.key}_delta_log_predictive_density_vs_independent"] = (
            comparison[f"{spec.key}_trial_log_predictive_density"] - baseline
        )
    return comparison.sort_index().reset_index(drop=True)


def _validate_detailed(frame: pd.DataFrame, *, expected_rows: int, hmm: bool) -> dict[str, Any]:
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} trial rows, found {len(frame)}")
    if frame["analysis_row_index"].nunique() != expected_rows:
        raise RuntimeError("Analysis trial keys are not unique")
    finite_columns = [
        "trial_log_predictive_density",
        "trial_predictive_density",
        *[f"emission_log_density_{state}" for state in STATE_KEYS],
        *[f"emission_density_{state}" for state in STATE_KEYS],
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("A required likelihood or density field is non-finite")
    filtered_sum = frame[[f"filtered_p_{state}" for state in STATE_KEYS]].sum(axis=1)
    predictive_sum = frame[[f"predictive_p_{state}" for state in STATE_KEYS]].sum(axis=1)
    result = {
        "rows": len(frame),
        "unique_trials": int(frame["analysis_row_index"].nunique()),
        "max_filtered_probability_error": float(np.max(np.abs(filtered_sum - 1.0))),
        "max_predictive_probability_error": float(np.max(np.abs(predictive_sum - 1.0))),
    }
    if result["max_filtered_probability_error"] > 1e-10:
        raise RuntimeError("Filtered state probabilities do not sum to one")
    if result["max_predictive_probability_error"] > 1e-10:
        raise RuntimeError("Predictive state probabilities do not sum to one")
    starts = frame["is_sequence_start"].astype(bool)
    previous_columns = [column for column in frame if column.startswith("previous_")]
    if previous_columns and not frame.loc[starts, previous_columns].isna().all().all():
        raise RuntimeError("A previous-trial field crosses a sequence boundary")
    if hmm:
        if frame.loc[starts, [*TRANSITION_COLUMNS, *XI_COLUMNS]].notna().any().any():
            raise RuntimeError("HMM transition fields must be blank at sequence starts")
        applicable = ~starts
        transition_error = 0.0
        for source in STATE_KEYS:
            columns = [f"transition_p_{source}_to_{target}" for target in STATE_KEYS]
            row_sum = frame.loc[applicable, columns].sum(axis=1)
            transition_error = max(transition_error, float(np.max(np.abs(row_sum - 1.0))))
        xi_sum = frame.loc[applicable, list(XI_COLUMNS)].sum(axis=1)
        xi_error = float(np.max(np.abs(xi_sum - 1.0)))
        result["max_transition_row_probability_error"] = transition_error
        result["max_smoothed_transition_probability_error"] = xi_error
        if transition_error > 1e-10 or xi_error > 1e-10:
            raise RuntimeError("HMM transition probabilities do not normalize")
    return result


def _excluded_trials(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw.insert(0, "raw_row_index", np.arange(len(raw), dtype=int))
    required_candidates = {
        "estimate_x": ("estimate_x", "resp_x", "response_x"),
        "estimate_y": ("estimate_y", "resp_y", "response_y"),
        "motion_direction": ("motion_direction", "direction", "stim_direction", "true_direction"),
        "motion_coherence": ("motion_coherence", "coherence"),
        "prior_std": ("prior_std", "prior_sd", "std_prior"),
        "prior_mean": ("prior_mean", "prior_mu", "mean_prior"),
    }
    resolved: dict[str, str] = {}
    for canonical, candidates in required_candidates.items():
        try:
            resolved[canonical] = next(column for column in candidates if column in raw)
        except StopIteration as exc:
            raise KeyError(f"Cannot audit exclusions: missing required field {canonical}") from exc
    missing = pd.DataFrame({
        canonical: raw[column].isna()
        for canonical, column in resolved.items()
    })
    excluded_mask = missing.any(axis=1)
    excluded = raw.loc[excluded_mask].copy()
    excluded["exclusion_reason"] = [
        ";".join(f"missing_{name}" for name, is_missing in row.items() if is_missing)
        for row in missing.loc[excluded_mask].to_dict("records")
    ]
    return excluded.reset_index(drop=True)


def _column_description(column: str) -> tuple[str, str, str]:
    exact = {
        "raw_row_index": ("Zero-based row position in the unchanged source CSV.", "row", "source CSV"),
        "analysis_row_index": ("Zero-based row position after exclusions and analysis sorting.", "row", "data loader"),
        "seq_id": ("Run-level sequence identifier used for fitting and cross-validation.", "sequence", "data loader"),
        "subject_id": ("Participant identifier supplied by the experiment dataset.", "identifier", "source CSV"),
        "session_id": ("Experimental session identifier.", "identifier", "source CSV"),
        "run_id": ("Run identifier; a model sequence never crosses this boundary.", "identifier", "source CSV"),
        "block_id": ("Alias of run_id retained for block-oriented analyses.", "identifier", "derived"),
        "trial_index": ("Trial index recorded by the experiment.", "trial", "source CSV"),
        "sequence_trial_index": ("Zero-based position within the run sequence.", "trial", "derived"),
        "sequence_length": ("Number of usable trials in the run sequence.", "trials", "derived"),
        "trial_time": ("Trial timing value recorded in the source data.", "source units", "source CSV"),
        "is_sequence_start": ("True only for the first usable row of a run.", "boolean", "derived"),
        "motion_direction_deg": ("True motion direction presented on the trial.", "degrees", "source CSV"),
        "motion_coherence": ("Objective random-dot motion coherence condition.", "proportion", "source CSV"),
        "prior_mean_deg": ("Center of the block's learned direction prior.", "degrees", "source CSV"),
        "prior_std_deg": ("Experimental prior-width condition.", "degrees", "source CSV"),
        "estimated_direction_deg": ("Participant's reported direction reconstructed from response coordinates.", "degrees", "derived"),
        "signed_stimulus_error_deg": ("Wrapped response minus true direction in [-180, 180] degrees.", "degrees", "derived"),
        "absolute_stimulus_error_deg": ("Absolute wrapped response error from the true direction.", "degrees", "derived"),
        "response_minus_prior_deg": ("Wrapped response minus prior center.", "degrees", "derived"),
        "stimulus_prior_conflict_deg": ("Absolute wrapped distance between true direction and prior center.", "degrees", "derived"),
        "model_key": ("Stable machine-readable model identifier.", "identifier", "exporter"),
        "model": ("Model label used in the analysis result tables.", "label", "exporter"),
        "fit_scope": ("final for all-data descriptive scores or oof for held-out sequence scores.", "category", "exporter"),
        "fold": ("Cross-validation fold for OOF rows; blank for final fits.", "fold", "exporter"),
        "relative_checkpoint_path": ("Checkpoint supplying every fitted parameter on the row.", "path", "checkpoint"),
        "run_key": ("Data-and-configuration compatibility key stored with the checkpoint.", "hash", "run manifest"),
        "trial_predictive_density": ("Predictive density assigned to the observed response, per radian.", "rad^-1", "model score"),
        "trial_log_predictive_density": ("Natural log of trial_predictive_density; for HMMs this is the forward-filter increment.", "nats", "model score"),
        "sequence_cumulative_log_likelihood": ("Cumulative sum of trial log predictive densities within the run.", "nats", "model score"),
        "sequence_log_likelihood": ("Sum of all trial log predictive densities in the run.", "nats", "model score"),
        "predictive_circular_mean_deg": ("Circular mean of the pre-response predictive mixture.", "degrees", "model prediction"),
        "predictive_resultant_length": ("Length of the predictive distribution's first circular moment.", "0 to 1", "model prediction"),
        "predictive_circular_sd_deg": ("Circular standard deviation derived from predictive resultant length.", "degrees", "model prediction"),
        "state_map_filtered": ("Most probable state after observing this trial and all earlier trials only.", "state", "online filtering"),
        "state_map_smoothed": ("Most probable state using the complete run, including later responses.", "state", "retrospective smoothing"),
        "transition_applies": ("Whether a within-run state transition enters the current row.", "boolean", "model structure"),
    }
    if column in exact:
        return exact[column]
    if column.startswith("previous_"):
        return ("Previous usable trial's corresponding value; blank at run starts.", "see field name", "derived")
    if column.startswith("emission_log_density_"):
        return ("Log response density under the named emission alone.", "log rad^-1", "emission model")
    if column.startswith("emission_density_"):
        return ("Response density under the named emission alone.", "rad^-1", "emission model")
    if column.startswith("predictive_p_"):
        return ("State/component probability before observing the current response.", "probability", "online prediction")
    if column.startswith("filtered_p_"):
        return ("State/component posterior after the current response, using no later responses.", "probability", "online filtering")
    if column.startswith("smoothed_p_"):
        return ("State posterior using the complete run, including later responses.", "probability", "retrospective smoothing")
    if column.startswith("state_log_contribution_"):
        return ("Log of predictive state probability times its emission density.", "log rad^-1", "model score")
    if column.startswith("transition_p_"):
        return ("State-transition probability into the current trial; blank at run starts.", "probability", "transition model")
    if column.startswith("smoothed_transition_posterior_"):
        return ("Whole-run posterior probability for the named transition into this trial.", "probability", "retrospective smoothing")
    if column.startswith("transition_covariate_"):
        return ("Raw or training-standardized transition covariate used on this trial.", "field-specific", "covariate HMM input")
    if column.startswith("transition_scaler_"):
        return ("Training-sample mean or SD used to standardize the named covariate.", "field-specific", "covariate HMM preprocessing")
    if column.startswith("mixture_weight_") or column.startswith("initial_p_"):
        return ("Fitted component weight or run-initial state probability.", "probability", "checkpoint parameter")
    if column.startswith("sensory_kappa") or column.startswith("prior_kappa"):
        return ("Applied von Mises concentration selected by coherence or prior width.", "kappa", "checkpoint parameter")
    if column.endswith("_delta_log_predictive_density_vs_independent"):
        return ("Trial log predictive density minus Independent Switching on the same trial.", "nats", "derived comparison")
    return ("Export field; interpret with its source table and model applicability.", "field-specific", "export package")


def _data_dictionary(column_files: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for column in sorted(column_files):
        meaning, unit, source = _column_description(column)
        rows.append({
            "column": column,
            "meaning": meaning,
            "unit_or_scale": unit,
            "source_or_timing": source,
            "appears_in": ";".join(sorted(column_files[column])),
        })
    return pd.DataFrame(rows)


def _readme_text() -> str:
    return """# Trial-level model exports

This directory is a deterministic scoring export from completed checkpoints. No model was fitted or updated while producing these files.

## Scopes

- `final/` uses all-data fitted parameters. These rows are descriptive, in-sample reconstructions and are not unbiased model-comparison estimates.
- `oof/` assigns every trial to exactly one held-out run-sequence fold. These log predictive densities are the authoritative trial-level model-comparison scores.
- `final/subject_hmm_trials.csv` uses each participant's own final checkpoint. Participant-level CV checkpoints do not exist, so there is no subject-HMM OOF file.

For an HMM, `trial_log_predictive_density` is the forward-filter likelihood increment. Its sum within a run exactly reproduces that run's likelihood under the saved parameters. `filtered_*` fields use current and past responses only. `smoothed_*` and `smoothed_transition_posterior_*` use the complete run and must not be interpreted as online predictions.

The objective coherence and prior-width columns are experimental design variables used to select fitted emission concentrations. They are not claims that participants explicitly knew numeric condition labels. Covariate-HMM OOF standardization is recomputed from training sequences only.

The fitted serial implementation prevents run-boundary carryover by substituting current-trial values when no previous trial exists. Consequently, response-serial start rows can use the current response in their center shift. `serial_sequence_start_current_trial_fallback` exposes those rows; they are preserved to reproduce the authoritative checkpoints and are a limitation of this baseline.

`trial_model_comparison.csv` places model scores side by side and reports differences from Independent Switching. It deliberately does not name a per-trial winner; model adequacy is assessed over held-out sequences and subjects.

Basic Bayesian and original condition-dependent Switching observers are contextual models in the report but are unavailable here because this authoritative run has no fitted checkpoints for them.

All angles are in degrees unless a density explicitly says `rad^-1`. Log likelihoods use natural logarithms (nats). Blank fields mean not applicable or unavailable, not zero.
"""


def _register_file(
    registry: list[dict[str, Any]],
    export_dir: Path,
    path: Path,
    frame: pd.DataFrame | None,
    category: str,
) -> None:
    registry.append({
        "path": path.relative_to(export_dir).as_posix(),
        "category": category,
        "rows": None if frame is None else int(len(frame)),
        "columns": None if frame is None else int(len(frame.columns)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    })


def export_trial_results(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    scope: Literal["final", "oof", "both"] = "both",
    include_subject: bool = True,
    overwrite: bool = False,
    allow_running_manifest: bool = False,
    update_parent_manifest: bool = True,
) -> dict[str, Any]:
    """Export checkpoint-based trial scores without invoking any fit function."""
    csv_path = Path(csv_path).resolve()
    out_dir = Path(out_dir).resolve()
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("status") != "complete" and not allow_running_manifest:
        raise RuntimeError("Trial export requires a completed parent run manifest")
    if parent_manifest.get("data_sha256") != sha256_file(csv_path):
        raise RuntimeError("Dataset hash does not match the completed run manifest")
    run_key = str(parent_manifest.get("run_key", ""))
    if not run_key:
        raise RuntimeError("Run manifest has no compatibility run key")
    required_stages = ["data", "cross_validation", "final_models"]
    if include_subject:
        required_stages.append("subject_models")
    incomplete = [
        stage
        for stage in required_stages
        if parent_manifest.get("stages", {}).get(stage, {}).get("status") != "complete"
    ]
    if incomplete:
        raise RuntimeError(f"Required analysis stages are incomplete: {', '.join(incomplete)}")

    requested_scopes = ("final", "oof") if scope == "both" else (scope,)
    export_dir = out_dir / "trial_exports"
    expected_targets = [
        export_dir / "README.md",
        export_dir / "export_manifest.json",
        export_dir / "data_dictionary.csv",
        export_dir / "excluded_trials.csv",
        export_dir / "model_fit_metadata.csv",
        export_dir / "model_parameters_long.csv",
    ]
    for requested_scope in requested_scopes:
        expected_targets.extend(
            export_dir / requested_scope / f"{spec.key}_trials.csv"
            for spec in GLOBAL_MODELS
        )
        expected_targets.append(export_dir / requested_scope / "trial_model_comparison.csv")
    if include_subject and "final" in requested_scopes:
        expected_targets.append(export_dir / "final" / "subject_hmm_trials.csv")
    existing = [path for path in expected_targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Trial export files already exist (for example {existing[0]}); pass overwrite=True to replace known files"
        )
    export_dir.mkdir(parents=True, exist_ok=True)

    data = load_direction_data(csv_path)
    expected_rows = int(parent_manifest.get("data", {}).get("usable_trials", len(data.df)))
    if len(data.df) != expected_rows:
        raise RuntimeError(f"Loaded {len(data.df)} usable trials; manifest records {expected_rows}")
    base = build_trial_base(data)
    all_seq = np.arange(len(data.sequences), dtype=int)
    file_registry: list[dict[str, Any]] = []
    column_files: dict[str, set[str]] = {}
    metadata_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"detailed_files": {}, "likelihood_reconciliation": {}}

    readme_path = export_dir / "README.md"
    temporary_readme = readme_path.with_suffix(".md.tmp")
    temporary_readme.write_text(_readme_text(), encoding="utf-8", newline="\n")
    os.replace(temporary_readme, readme_path)
    _register_file(file_registry, export_dir, readme_path, None, "documentation")

    excluded = _excluded_trials(csv_path)
    if len(excluded) != data.dropped_n_rows:
        raise RuntimeError("Excluded-trial audit does not match the data loader's dropped row count")
    excluded_path = _atomic_write_csv(excluded, export_dir / "excluded_trials.csv")
    _register_file(file_registry, export_dir, excluded_path, excluded, "audit")
    for column in excluded.columns:
        column_files.setdefault(column, set()).add("excluded_trials.csv")

    model_info = pd.read_csv(out_dir / "model_info_criteria.csv").set_index("model")
    compact_scopes: dict[str, dict[str, pd.DataFrame]] = {}
    fold_by_row: pd.Series | None = None
    oof_sequence_rows: list[pd.DataFrame] = []

    if "final" in requested_scopes:
        compact_scopes["final"] = {}
        for spec in GLOBAL_MODELS:
            fit, checkpoint_path = _load_required_fit(out_dir, spec.final_checkpoint, run_key)
            model_data = data
            scaler_provenance = (
                "all_usable_trials_standardization"
                if spec.family == "covariate_hmm"
                else "not_applicable"
            )
            score = _score_model(spec, model_data, fit, all_seq).sort_index()
            scored_ll = float(score["trial_log_predictive_density"].sum())
            detailed = _make_detailed_frame(
                base,
                score,
                spec=spec,
                scope="final",
                fold=None,
                fit=fit,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                run_key=run_key,
                parameter_source="all_usable_trials_final_fit",
                scaler_provenance=scaler_provenance,
                training_sequence_count=len(all_seq),
                training_trial_count=len(data.df),
            )
            file_name = f"final/{spec.key}_trials.csv"
            validation["detailed_files"][file_name] = _validate_detailed(
                detailed,
                expected_rows=expected_rows,
                hmm=spec.family in {"hmm", "covariate_hmm"},
            )
            path = _atomic_write_csv(detailed, export_dir / file_name)
            _register_file(file_registry, export_dir, path, detailed, "final_trial_scores")
            for column in detailed.columns:
                column_files.setdefault(column, set()).add(file_name)
            compact_scopes["final"][spec.key] = _compact_scores(score, spec.key)
            metadata_rows.append(_fit_metadata_row(
                spec,
                fit,
                scope="final",
                fold=None,
                subject_id=np.nan,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                scored_sequence_count=len(all_seq),
                scored_trial_count=len(data.df),
                training_sequence_count=len(all_seq),
                training_trial_count=len(data.df),
                recomputed_scored_log_likelihood=scored_ll,
                scaler_provenance=scaler_provenance,
            ))
            parameter_rows.extend(_parameter_rows(
                spec,
                fit,
                model_data,
                scope="final",
                fold=None,
                subject_id=np.nan,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                scaler_provenance=scaler_provenance,
            ))
            recorded = float(model_info.loc[spec.model_name, "train_ll"])
            difference = scored_ll - recorded
            if abs(difference) > 0.05:
                raise RuntimeError(
                    f"Final {spec.model_name} trial scores differ from model_info_criteria by {difference}"
                )
            validation["likelihood_reconciliation"][f"final/{spec.key}"] = {
                "trial_score_sum": scored_ll,
                "recorded_model_info_log_likelihood": recorded,
                "difference": difference,
                "tolerance": 0.05,
            }
            print(f"Wrote {file_name}: {len(detailed):,} rows", flush=True)
            del detailed

        comparison = _comparison_frame(base, compact_scopes["final"], None, "final")
        comparison_path = _atomic_write_csv(
            comparison, export_dir / "final" / "trial_model_comparison.csv"
        )
        _register_file(file_registry, export_dir, comparison_path, comparison, "final_comparison")
        for column in comparison.columns:
            column_files.setdefault(column, set()).add("final/trial_model_comparison.csv")
        del comparison

    if "oof" in requested_scopes:
        config = parent_manifest.get("resolved_config", {})
        n_splits = int(config.get("cv", {}).get("n_splits", 4))
        split_seed = int(config.get("diagnostics", {}).get("seed", config.get("cv", {}).get("seed", 42)))
        splits = list(make_cv_splits(data, n_splits=n_splits, seed=split_seed))
        test_seq_seen = np.zeros(len(data.sequences), dtype=int)
        fold_values = np.full(len(data.df), -1, dtype=int)
        for fold, (_, test_seq) in enumerate(splits, start=1):
            test_seq_seen[test_seq] += 1
            fold_values[_selected_indices(data, test_seq)] = fold
        if not np.all(test_seq_seen == 1) or np.any(fold_values < 1):
            raise RuntimeError("Reconstructed CV folds do not score each sequence and trial exactly once")
        fold_by_row = pd.Series(fold_values, index=np.arange(len(data.df)), name="fold")
        compact_scopes["oof"] = {}

        for spec in GLOBAL_MODELS:
            score_parts: list[pd.DataFrame] = []
            detailed_parts: list[pd.DataFrame] = []
            for fold, (train_seq, test_seq) in enumerate(splits, start=1):
                checkpoint_name = spec.cv_checkpoint.format(fold=fold)
                fit, checkpoint_path = _load_required_fit(out_dir, checkpoint_name, run_key)
                if spec.family == "covariate_hmm":
                    means, sds = transition_scaler_from_sequences(data, train_seq)
                    model_data = with_transition_scaler(data, means, sds)
                    scaler_provenance = f"fold_{fold}_training_sequences_only"
                else:
                    model_data = data
                    scaler_provenance = "not_applicable"
                score = _score_model(spec, model_data, fit, test_seq).sort_index()
                score_parts.append(score)
                training_trial_count = int(sum(len(data.sequences[int(sid)]) for sid in train_seq))
                detailed_parts.append(_make_detailed_frame(
                    base,
                    score,
                    spec=spec,
                    scope="oof",
                    fold=fold,
                    fit=fit,
                    checkpoint_path=checkpoint_path,
                    out_dir=out_dir,
                    run_key=run_key,
                    parameter_source=f"fold_{fold}_training_sequences_excluding_current_run",
                    scaler_provenance=scaler_provenance,
                    training_sequence_count=len(train_seq),
                    training_trial_count=training_trial_count,
                ))
                scored_ll = float(score["trial_log_predictive_density"].sum())
                metadata_rows.append(_fit_metadata_row(
                    spec,
                    fit,
                    scope="oof",
                    fold=fold,
                    subject_id=np.nan,
                    checkpoint_path=checkpoint_path,
                    out_dir=out_dir,
                    scored_sequence_count=len(test_seq),
                    scored_trial_count=len(score),
                    training_sequence_count=len(train_seq),
                    training_trial_count=training_trial_count,
                    recomputed_scored_log_likelihood=scored_ll,
                    scaler_provenance=scaler_provenance,
                ))
                parameter_rows.extend(_parameter_rows(
                    spec,
                    fit,
                    model_data,
                    scope="oof",
                    fold=fold,
                    subject_id=np.nan,
                    checkpoint_path=checkpoint_path,
                    out_dir=out_dir,
                    scaler_provenance=scaler_provenance,
                ))
                seq_ids_for_rows = base.set_index("analysis_row_index").loc[score.index, "seq_id"].to_numpy()
                sequence_totals = pd.DataFrame({
                    "seq_id": seq_ids_for_rows,
                    "trial_log_predictive_density": score["trial_log_predictive_density"].to_numpy(),
                }).groupby("seq_id", as_index=False)["trial_log_predictive_density"].sum()
                sequence_totals.insert(0, "model", spec.model_name)
                sequence_totals.insert(0, "fold", fold)
                oof_sequence_rows.append(sequence_totals)

            score_all = pd.concat(score_parts).sort_index()
            detailed = pd.concat(detailed_parts, ignore_index=True).sort_values("analysis_row_index").reset_index(drop=True)
            file_name = f"oof/{spec.key}_trials.csv"
            validation["detailed_files"][file_name] = _validate_detailed(
                detailed,
                expected_rows=expected_rows,
                hmm=spec.family in {"hmm", "covariate_hmm"},
            )
            path = _atomic_write_csv(detailed, export_dir / file_name)
            _register_file(file_registry, export_dir, path, detailed, "oof_trial_scores")
            for column in detailed.columns:
                column_files.setdefault(column, set()).add(file_name)
            compact_scopes["oof"][spec.key] = _compact_scores(score_all, spec.key)
            print(f"Wrote {file_name}: {len(detailed):,} rows", flush=True)
            del detailed, detailed_parts, score_parts

        comparison = _comparison_frame(base, compact_scopes["oof"], fold_by_row, "oof")
        comparison_path = _atomic_write_csv(
            comparison, export_dir / "oof" / "trial_model_comparison.csv"
        )
        _register_file(file_registry, export_dir, comparison_path, comparison, "oof_comparison")
        for column in comparison.columns:
            column_files.setdefault(column, set()).add("oof/trial_model_comparison.csv")
        del comparison

        exported_sequence = pd.concat(oof_sequence_rows, ignore_index=True)
        reference_sequence = pd.read_csv(out_dir / "per_sequence_cv_results.csv")[
            ["fold", "model", "seq_id", "test_ll"]
        ]
        reconciled = reference_sequence.merge(
            exported_sequence,
            on=["fold", "model", "seq_id"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not (reconciled["_merge"] == "both").all():
            raise RuntimeError("OOF trial scores do not cover the existing per-sequence CV table exactly")
        differences = reconciled["trial_log_predictive_density"] - reconciled["test_ll"]
        max_abs_difference = float(np.max(np.abs(differences)))
        if max_abs_difference > 1e-7:
            raise RuntimeError(
                f"OOF trial score sums do not reproduce per-sequence CV likelihoods: {max_abs_difference}"
            )
        validation["likelihood_reconciliation"]["oof_per_sequence"] = {
            "sequences_by_model": int(len(reconciled)),
            "max_absolute_difference": max_abs_difference,
            "tolerance": 1e-7,
        }

    if include_subject and "final" in requested_scopes:
        subject_spec = ModelSpec(
            "subject_hmm",
            "Subject_specific_HMM_static",
            "hmm",
            "subject_{subject}_hmm",
            "",
        )
        subject_parts: list[pd.DataFrame] = []
        for subject in data.subject_values:
            checkpoint_name = subject_spec.final_checkpoint.format(subject=int(subject))
            fit, checkpoint_path = _load_required_fit(out_dir, checkpoint_name, run_key)
            subject_seq = data.seq_meta.index[data.seq_meta["subject_id"] == subject].to_numpy(dtype=int)
            score = score_hmm(data, fit.params, subject_seq, covariate=False).sort_index()
            subject_trial_count = len(score)
            subject_parts.append(_make_detailed_frame(
                base,
                score,
                spec=subject_spec,
                scope="final",
                fold=None,
                fit=fit,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                run_key=run_key,
                parameter_source=f"subject_{int(subject)}_all_trials_final_fit",
                scaler_provenance="not_applicable",
                training_sequence_count=len(subject_seq),
                training_trial_count=subject_trial_count,
                fit_subject_id=subject,
            ))
            scored_ll = float(score["trial_log_predictive_density"].sum())
            metadata_rows.append(_fit_metadata_row(
                subject_spec,
                fit,
                scope="final",
                fold=None,
                subject_id=subject,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                scored_sequence_count=len(subject_seq),
                scored_trial_count=subject_trial_count,
                training_sequence_count=len(subject_seq),
                training_trial_count=subject_trial_count,
                recomputed_scored_log_likelihood=scored_ll,
                scaler_provenance="not_applicable",
            ))
            parameter_rows.extend(_parameter_rows(
                subject_spec,
                fit,
                data,
                scope="final",
                fold=None,
                subject_id=subject,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                scaler_provenance="not_applicable",
            ))
        subject_detailed = pd.concat(subject_parts, ignore_index=True).sort_values("analysis_row_index").reset_index(drop=True)
        file_name = "final/subject_hmm_trials.csv"
        validation["detailed_files"][file_name] = _validate_detailed(
            subject_detailed,
            expected_rows=expected_rows,
            hmm=True,
        )
        subject_path = _atomic_write_csv(subject_detailed, export_dir / file_name)
        _register_file(file_registry, export_dir, subject_path, subject_detailed, "subject_final_trial_scores")
        for column in subject_detailed.columns:
            column_files.setdefault(column, set()).add(file_name)
        print(f"Wrote {file_name}: {len(subject_detailed):,} rows", flush=True)

    metadata = pd.DataFrame(metadata_rows).sort_values(
        ["fit_scope", "model_key", "fold", "subject_id"], na_position="first"
    ).reset_index(drop=True)
    metadata_path = _atomic_write_csv(metadata, export_dir / "model_fit_metadata.csv")
    _register_file(file_registry, export_dir, metadata_path, metadata, "fit_metadata")
    for column in metadata.columns:
        column_files.setdefault(column, set()).add("model_fit_metadata.csv")

    parameters = pd.DataFrame(parameter_rows).sort_values(
        ["fit_scope", "model_key", "fold", "subject_id", "parameter_group", "level_1", "level_2", "covariate"],
        na_position="first",
    ).reset_index(drop=True)
    parameters_path = _atomic_write_csv(parameters, export_dir / "model_parameters_long.csv")
    _register_file(file_registry, export_dir, parameters_path, parameters, "parameters")
    for column in parameters.columns:
        column_files.setdefault(column, set()).add("model_parameters_long.csv")

    dictionary = _data_dictionary(column_files)
    dictionary_path = _atomic_write_csv(dictionary, export_dir / "data_dictionary.csv")
    _register_file(file_registry, export_dir, dictionary_path, dictionary, "documentation")

    export_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_at": parent_manifest.get("completed_at"),
        "fit_independent": True,
        "parent_run_key": run_key,
        "parent_config_hash": parent_manifest.get("config_hash"),
        "data_sha256": parent_manifest.get("data_sha256"),
        "csv_path": str(csv_path),
        "analysis_out_dir": str(out_dir),
        "trial_export_dir": str(export_dir),
        "requested_scope": scope,
        "include_subject": include_subject,
        "usable_trials": len(data.df),
        "excluded_trials": len(excluded),
        "sequences": len(data.sequences),
        "global_models": [spec.key for spec in GLOBAL_MODELS],
        "unavailable_contextual_models": [
            {
                "model": "Basic Bayesian observer",
                "reason": "No authoritative fitted checkpoint was included in this run.",
            },
            {
                "model": "Original condition-dependent Switching observer",
                "reason": "No authoritative fitted checkpoint was included in this run.",
            },
        ],
        "files": sorted(file_registry, key=lambda row: row["path"]),
        "validation": validation,
    }
    atomic_write_json(export_manifest, export_dir / "export_manifest.json")

    if update_parent_manifest:
        parent_manifest.setdefault("stages", {})["trial_exports"] = {
            "status": "complete",
            "updated_at": utc_now(),
            "scopes": list(requested_scopes),
            "global_models": len(GLOBAL_MODELS),
            "subject_model": bool(include_subject and "final" in requested_scopes),
            "usable_trials": len(data.df),
            "manifest": str((export_dir / "export_manifest.json").resolve()),
        }
        atomic_write_json(parent_manifest, manifest_path)
    return export_manifest
