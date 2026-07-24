from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from perceptual_arbitration.checkpoints import save_fit_checkpoint
from perceptual_arbitration.circular import circ_absdiff
from perceptual_arbitration.covariate_hmm import (
    CovariateHMMFit,
    forward_backward_timevarying,
    init_covariate_hmm,
    transition_logA_sequence,
    transition_logA_t,
)
from perceptual_arbitration.data import load_direction_data
from perceptual_arbitration.diagnostics import (
    ppc_histogram_intervals,
    ppc_metric_intervals,
    simulate_covariate_hmm_draw,
    simulate_static_hmm_draw,
)
from perceptual_arbitration.hmm import (
    HMMFit,
    HMMParams,
    STATE_NAMES,
    emission_logB_stable,
    forward_backward_scaled,
)
from perceptual_arbitration.posterior_predictive import run_checkpoint_posterior_predictive_checks
from perceptual_arbitration.run_metadata import atomic_write_json, sha256_file
from perceptual_arbitration.sp_posterior_predictive import (
    SP_METHOD,
    _response_bundle,
    _retention_summary,
    decode_smoothed_marginal_states,
    run_lapse_excluded_posterior_predictive_checks,
)


def _xy(angle_deg: float) -> tuple[float, float]:
    angle = np.deg2rad(angle_deg)
    return float(np.cos(angle)), float(np.sin(angle))


def _write_csv(path: Path) -> None:
    rows = []
    for run in [1, 2, 3]:
        for trial in range(1, 10):
            direction = 190 + 13 * trial
            estimate = direction + (9 if trial % 2 else -7)
            x, y = _xy(estimate)
            rows.append({
                "trial_index": trial,
                "trial_time": trial * 1.5,
                "response_arrow_start_angle": 0.0,
                "motion_direction": direction,
                "motion_coherence": [0.06, 0.12, 0.24][(trial - 1) % 3],
                "estimate_x": x,
                "estimate_y": y,
                "reaction_time": 0.7,
                "raw_response_time": 0.8,
                "prior_std": [10, 20, 40][run - 1],
                "prior_mean": 225,
                "subject_id": 1,
                "experiment_name": "synthetic",
                "experiment_id": 99,
                "session_id": 1,
                "run_id": run,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _parameters(data):
    kappa_s = np.array([2.0, 5.0, 12.0])
    kappa_p = np.array([18.0, 8.0, 3.0])
    static = HMMParams(
        pi=np.array([0.45, 0.35, 0.20]),
        A=np.array([[0.65, 0.25, 0.10], [0.20, 0.65, 0.15], [0.35, 0.25, 0.40]]),
        kappa_s=kappa_s,
        kappa_p=kappa_p,
    )
    covariate = init_covariate_hmm(data, seed=7)
    covariate.kappa_s = kappa_s.copy()
    covariate.kappa_p = kappa_p.copy()
    prev_error = data.transition_names.index("prev_error")
    covariate.B[0, 0, prev_error] = 0.8
    covariate.B[0, 1, prev_error] = -0.4
    covariate.B[1, 1, prev_error] = 0.5
    return static, covariate


def test_covariate_simulation_uses_recursive_simulated_error_and_fitted_scaler(tmp_path):
    csv_path = tmp_path / "trials.csv"
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    _, params = _parameters(data)

    first = simulate_covariate_hmm_draw(data, params, simulation=3, seed=42)
    second = simulate_covariate_hmm_draw(data, params, simulation=3, seed=42)
    pd.testing.assert_frame_equal(first, second)

    state_index = {state: index for index, state in enumerate(STATE_NAMES)}
    raw_prev_error_index = data.transition_names.index("prev_error") - 1
    starts = first["is_sequence_start"].to_numpy(dtype=bool)
    assert first.loc[starts, ["transition_prev_error_deg", "transition_probability_used"]].isna().all().all()
    assert np.allclose(first.loc[~starts, "transition_probability_row_sum"], 1.0)

    simulated = np.deg2rad(first["estimate_deg"].to_numpy())
    for idx in data.sequences:
        for previous_row, row in zip(idx[:-1], idx[1:]):
            expected_error = float(circ_absdiff(simulated[previous_row], data.theta[previous_row]))
            assert np.isclose(first.loc[row, "transition_prev_error_deg"], np.rad2deg(expected_error))
            raw = data.X_transition_raw[row].copy()
            raw[raw_prev_error_index] = expected_error
            standardized = np.r_[1.0, (raw - data.transition_means) / data.transition_sds]
            transition = np.exp(transition_logA_t(params, standardized))
            expected_probability = transition[
                state_index[first.loc[previous_row, "state"]],
                state_index[first.loc[row, "state"]],
            ]
            assert np.isclose(first.loc[row, "transition_probability_used"], expected_probability)


def test_covariate_simulation_does_not_read_observed_responses(tmp_path):
    csv_path = tmp_path / "trials.csv"
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    _, params = _parameters(data)
    changed = replace(data, y=np.mod(data.y + 1.234, 2 * np.pi))
    original = simulate_covariate_hmm_draw(data, params, simulation=5, seed=42)
    altered = simulate_covariate_hmm_draw(changed, params, simulation=5, seed=42)
    pd.testing.assert_frame_equal(original, altered)


def test_static_simulation_is_deterministic_and_model_labelled(tmp_path):
    csv_path = tmp_path / "trials.csv"
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    static, _ = _parameters(data)
    first = simulate_static_hmm_draw(data, static, simulation=0, seed=42)
    second = simulate_static_hmm_draw(data, static, simulation=0, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["model"]) == {"HMM_static"}
    assert len(first) == len(data.df)


def test_ppc_interval_keys_are_stable_after_csv_float_round_trip():
    metrics = {
        "mean_abs_error_deg": 10.0,
        "median_abs_error_deg": 9.0,
        "mean_cos_error": 0.9,
        "circular_std_error_deg": 12.0,
        "prior_like_rate": 0.4,
    }
    summary = pd.DataFrame([
        {"source": "observed", "simulation": -1, "motion_coherence": 0.06, "prior_std": 10.0, **metrics},
        {"source": "simulated", "simulation": 0, "motion_coherence": 0.0599999999999999, "prior_std": 10.0, **metrics},
        {"source": "simulated", "simulation": 1, "motion_coherence": 0.0600000000000001, "prior_std": 10.0, **metrics},
    ])
    metric_intervals = ppc_metric_intervals(summary)
    assert len(metric_intervals) == 5
    assert set(metric_intervals["condition"]) == {"0.06/10"}

    histograms = pd.DataFrame([
        {"source": "observed", "simulation": -1, "motion_coherence": 0.06, "prior_std": 10.0, "reference": "stimulus", "bin_left_deg": -15.0, "bin_right_deg": 0.0, "proportion": 0.2},
        {"source": "simulated", "simulation": 0, "motion_coherence": 0.0599999999999999, "prior_std": 10.0, "reference": "stimulus", "bin_left_deg": -15.0, "bin_right_deg": 0.0, "proportion": 0.1},
        {"source": "simulated", "simulation": 1, "motion_coherence": 0.0600000000000001, "prior_std": 10.0, "reference": "stimulus", "bin_left_deg": -15.0, "bin_right_deg": 0.0, "proportion": 0.3},
    ])
    histogram_intervals = ppc_histogram_intervals(histograms)
    assert len(histogram_intervals) == 1
    assert np.isclose(histogram_intervals.iloc[0]["observed"], 0.2)

    retention = pd.DataFrame([
        {"source": "observed", "simulation": -1, "motion_coherence": 0.06, "prior_std": 10.0, "decoded_state": "S_sensory", "retained": True, "n": 80},
        {"source": "observed", "simulation": -1, "motion_coherence": 0.06, "prior_std": 10.0, "decoded_state": "L_lapse", "retained": False, "n": 20},
        {"source": "simulated", "simulation": 0, "motion_coherence": 0.0599999999999999, "prior_std": 10.0, "decoded_state": "S_sensory", "retained": True, "n": 75},
        {"source": "simulated", "simulation": 0, "motion_coherence": 0.0599999999999999, "prior_std": 10.0, "decoded_state": "L_lapse", "retained": False, "n": 25},
        {"source": "simulated", "simulation": 1, "motion_coherence": 0.0600000000000001, "prior_std": 10.0, "decoded_state": "S_sensory", "retained": True, "n": 85},
        {"source": "simulated", "simulation": 1, "motion_coherence": 0.0600000000000001, "prior_std": 10.0, "decoded_state": "L_lapse", "retained": False, "n": 15},
    ])
    retention_summary = _retention_summary(retention)
    assert len(retention_summary) == 1
    assert retention_summary.iloc[0]["observed_retained_n"] == 80
    assert np.isclose(retention_summary.iloc[0]["simulated_retained_rate_mean"], 0.8)


def test_symmetric_decoder_is_normalized_deterministic_and_response_driven(tmp_path):
    csv_path = tmp_path / "trials.csv"
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    static, covariate = _parameters(data)
    simulated = simulate_covariate_hmm_draw(data, covariate, simulation=2, seed=42)
    response = simulated["estimate_deg"].to_numpy(dtype=float)
    first, validation = decode_smoothed_marginal_states(
        data,
        covariate,
        response,
        model="covariate_hmm",
        recorded_prev_error_deg=simulated["transition_prev_error_deg"].to_numpy(dtype=float),
    )
    second, _ = decode_smoothed_marginal_states(
        replace(data, y=np.mod(data.y + 0.75, 2 * np.pi)),
        covariate,
        response,
        model="covariate_hmm",
        recorded_prev_error_deg=simulated["transition_prev_error_deg"].to_numpy(dtype=float),
    )
    pd.testing.assert_frame_equal(first, second)
    assert np.allclose(first[["p_sensory", "p_prior", "p_lapse"]].sum(axis=1), 1.0)
    assert set(first["state_map_method"]) == {SP_METHOD}
    assert validation["max_simulated_previous_error_error_rad"] < 1e-12

    sequence = data.sequences[0]
    simulated_bundle, _, _ = _response_bundle(data, response, covariate=True)
    covariate_emission_params = HMMParams(
        pi=covariate.pi,
        A=np.ones((3, 3), dtype=float) / 3.0,
        kappa_s=covariate.kappa_s,
        kappa_p=covariate.kappa_p,
    )
    log_b = emission_logB_stable(simulated_bundle, covariate_emission_params, sequence)
    log_a = transition_logA_sequence(covariate, simulated_bundle.X_transition[sequence])
    _, direct_covariate_gamma, _ = forward_backward_timevarying(
        log_b,
        np.log(np.maximum(covariate.pi, 1e-300)),
        log_a,
    )
    assert np.allclose(
        first.loc[sequence, ["p_sensory", "p_prior", "p_lapse"]],
        direct_covariate_gamma,
        atol=1e-11,
    )

    static_decoded, _ = decode_smoothed_marginal_states(
        data,
        static,
        data.df["estimate_deg"].to_numpy(dtype=float),
        model="static_hmm",
    )
    assert np.allclose(static_decoded[["p_sensory", "p_prior", "p_lapse"]].sum(axis=1), 1.0)
    static_log_b = emission_logB_stable(data, static, sequence)
    _, direct_static_gamma, _ = forward_backward_scaled(static_log_b, static.pi, static.A)
    assert np.allclose(
        static_decoded.loc[sequence, ["p_sensory", "p_prior", "p_lapse"]],
        direct_static_gamma,
        atol=1e-11,
    )


def test_checkpoint_runner_writes_both_models_and_preserves_checkpoints(tmp_path):
    csv_path = tmp_path / "trials.csv"
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    static, covariate = _parameters(data)
    run_key = "synthetic-run-key"
    static_checkpoint = out_dir / "checkpoints" / "final_hmm_static.joblib"
    covariate_checkpoint = out_dir / "checkpoints" / "final_covariate_hmm.joblib"
    save_fit_checkpoint(static_checkpoint, run_key, HMMFit(static, -1.0, 3, True, 11, [-1.0]))
    save_fit_checkpoint(covariate_checkpoint, run_key, CovariateHMMFit(covariate, -1.0, 3, True, 12, [-1.0]))
    checkpoint_hashes = (sha256_file(static_checkpoint), sha256_file(covariate_checkpoint))

    posterior_rows = []
    for seq_id, idx in enumerate(data.sequences):
        for position, row in enumerate(idx):
            posterior_rows.append({
                "row_index": int(row),
                "seq_id": seq_id,
                "subject_id": data.df.loc[row, "subject_id"],
                "session_id": data.df.loc[row, "session_id"],
                "run_id": data.df.loc[row, "run_id"],
                "trial_index": data.df.loc[row, "trial_index"],
                "state_map": STATE_NAMES[position % 3],
            })
    pd.DataFrame(posterior_rows).to_csv(out_dir / "posterior_states.csv", index=False)
    atomic_write_json({
        "status": "complete",
        "run_key": run_key,
        "data_sha256": sha256_file(csv_path),
        "resolved_config": {"diagnostics": {"n_ppc_simulations": 2}},
        "stages": {},
    }, out_dir / "run_manifest.json")

    manifest = run_checkpoint_posterior_predictive_checks(
        csv_path,
        out_dir,
        models=["static_hmm", "covariate_hmm"],
        n_simulations=4,
        seed=42,
        resume=True,
    )
    assert manifest["status"] == "complete"
    assert manifest["execution_override"]
    assert manifest["models"] == ["static_hmm", "covariate_hmm"]
    assert checkpoint_hashes == (sha256_file(static_checkpoint), sha256_file(covariate_checkpoint))

    for name in [
        "posterior_predictive_simulated_responses.csv",
        "covariate_hmm_posterior_predictive_simulated_responses.csv",
    ]:
        frame = pd.read_csv(out_dir / name)
        assert len(frame) == 4 * len(data.df)
        assert set(frame["simulation"]) == {0, 1, 2, 3}
        assert frame.groupby("simulation")["seq_id"].nunique().eq(len(data.sequences)).all()
    coverage = pd.read_csv(out_dir / "posterior_predictive_model_coverage.csv")
    assert set(coverage["model"]) == {"HMM_static", "Covariate_HMM"}
    assert set(coverage["metric"]) == {
        "mean_abs_error_deg", "median_abs_error_deg", "mean_cos_error",
        "circular_std_error_deg", "prior_like_rate",
    }
    parent = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert parent["run_key"] == run_key
    assert parent["stages"]["posterior_predictive_checks"]["simulations_per_model"] == 4

    static_decoded, _ = decode_smoothed_marginal_states(
        data,
        static,
        data.df["estimate_deg"].to_numpy(dtype=float),
        model="static_hmm",
    )
    static_decoded.to_csv(out_dir / "posterior_states.csv", index=False)
    sp_manifest = run_lapse_excluded_posterior_predictive_checks(
        csv_path,
        out_dir,
        models=["static_hmm", "covariate_hmm"],
        n_simulations=4,
        seed=42,
        resume=True,
    )
    assert sp_manifest["status"] == "complete"
    assert sp_manifest["response_simulation_reused"]
    assert sp_manifest["state_path_summary"] == SP_METHOD
    assert sp_manifest["raw_ppc_hashes_before"] == sp_manifest["raw_ppc_hashes_after"]
    sp_coverage = pd.read_csv(out_dir / "posterior_predictive_sp_model_coverage.csv")
    assert set(sp_coverage["model"]) == {"HMM_static", "Covariate_HMM"}
    assert set(sp_coverage["metric"]) == {
        "mean_abs_error_deg", "median_abs_error_deg", "mean_cos_error",
        "circular_std_error_deg", "prior_like_rate",
    }
    retention = pd.read_csv(out_dir / "posterior_predictive_sp_model_retention.csv")
    assert set(retention["decoded_state"]) == set(STATE_NAMES)
    expected_condition_n = int(data.df.groupby(["motion_coherence", "prior_std"]).size().iloc[0])
    assert retention.groupby(["model", "source", "simulation", "motion_coherence", "prior_std"])["n"].sum().eq(expected_condition_n).all()
