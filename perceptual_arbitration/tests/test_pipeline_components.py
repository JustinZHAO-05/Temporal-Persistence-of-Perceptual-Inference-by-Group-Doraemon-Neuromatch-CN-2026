from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from perceptual_arbitration.covariate_hmm import (
    _transition_objective_gradient,
    init_covariate_hmm,
    transition_logA_t,
)
from perceptual_arbitration.data import load_direction_data, transition_scaler_from_sequences, with_transition_scaler
from perceptual_arbitration.diagnostics import (
    behavioral_condition_summary,
    bootstrap_model_differences,
    conditional_covariate_effect_intervals,
    dataset_summary,
    posterior_state_occupancy,
    stationary_distribution,
)
from perceptual_arbitration.hmm import fit_hmm_em, fit_hmm_multistart, forward_backward_log
from perceptual_arbitration.independent_switching import fit_independent_em
from perceptual_arbitration.plotting import generate_figures
from perceptual_arbitration.publication import (
    _REPORT_BROWSER_STYLE_ID,
    _inject_report_browser_style,
    publication_status,
    result_claims,
)
from perceptual_arbitration.serial_dependence import previous_arrays


def _xy(angle_deg: float) -> tuple[float, float]:
    r = np.deg2rad(angle_deg)
    return float(np.cos(r)), float(np.sin(r))


def _write_small_csv(path):
    rows = []
    for subject in [1, 2]:
        for run in [1, 2]:
            for trial in range(1, 7):
                direction = 225 if trial % 2 else 235
                estimate = direction if trial <= 3 else 225
                x, y = _xy(estimate)
                rows.append({
                    "trial_index": trial,
                    "trial_time": trial * 2.0,
                    "response_arrow_start_angle": 0,
                    "motion_direction": direction,
                    "motion_coherence": [0.06, 0.12, 0.24][trial % 3],
                    "estimate_x": x,
                    "estimate_y": y,
                    "reaction_time": 1.0,
                    "raw_response_time": 1.0,
                    "prior_std": [10, 20][run - 1],
                    "prior_mean": 225,
                    "subject_id": subject,
                    "experiment_name": "test",
                    "experiment_id": 1,
                    "session_id": 1,
                    "run_id": run,
                })
    rows[3]["estimate_x"] = np.nan
    rows[3]["estimate_y"] = np.nan
    pd.DataFrame(rows).to_csv(path, index=False)


def test_loader_drops_missing_and_preserves_sequences(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    assert len(data.df) == 23
    assert len(data.sequences) == 4
    assert data.X_transition.shape[1] == len(data.transition_names)
    assert data.X_transition_raw.shape[1] == len(data.transition_names) - 1


def test_train_fold_transition_scaler_changes_scaling(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    means, sds = transition_scaler_from_sequences(data, [0, 1])
    scaled = with_transition_scaler(data, means, sds)
    train_idx = np.concatenate([scaled.sequences[0], scaled.sequences[1]])
    assert np.allclose(scaled.X_transition[train_idx, 1:].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scaled.X_transition[train_idx, 1:].std(axis=0), 1.0, atol=1e-12)


def test_forward_backward_normalizes():
    logB = np.log(np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]))
    logpi = np.log(np.array([0.5, 0.4, 0.1]))
    logA = np.log(np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.2, 0.5]]))
    ll, gamma, xi = forward_backward_log(logB, logpi, logA)
    assert np.isfinite(ll)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert xi.shape == (2, 3, 3)


def test_em_histories_are_finite_and_monotone(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    seq_ids = np.arange(len(data.sequences))
    hmm = fit_hmm_em(data, seq_ids, seed=1, max_iter=6, min_iter=1)
    ind = fit_independent_em(data, seq_ids, seed=2, max_iter=6, min_iter=1)
    assert np.all(np.isfinite(hmm.history))
    assert np.all(np.diff(hmm.history) >= -1e-5)
    assert np.all(np.isfinite(ind.history))
    assert np.all(np.diff(ind.history) >= -1e-5)


def test_serial_previous_arrays_do_not_cross_sequences(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    prev_theta, prev_y = previous_arrays(data)
    for idx in data.sequences:
        assert prev_theta[idx[0]] == data.theta[idx[0]]
        assert prev_y[idx[0]] == data.y[idx[0]]


def test_covariate_transition_rows_sum_to_one(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    params = init_covariate_hmm(data, seed=3)
    A = np.exp(transition_logA_t(params, data.X_transition[0]))
    assert np.allclose(A.sum(axis=1), 1.0)


def test_chunked_transition_objective_matches_dense_reference():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(97, 5))
    Xi = rng.gamma(1.5, size=(97, 3))
    v = rng.normal(scale=0.2, size=10)
    l2 = 0.7

    B = np.zeros((3, 5))
    B[:-1] = v.reshape(2, 5)
    logits = X @ B.T
    logits -= logsumexp(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    expected_nll = -np.sum(Xi * logits) + 0.5 * l2 * np.sum(B[:-1] ** 2)
    residual = probs * Xi.sum(axis=1, keepdims=True) - Xi
    expected_grad = residual.T @ X
    expected_grad[:-1] += l2 * B[:-1]

    nll, grad = _transition_objective_gradient(v, X, Xi, l2, block_size=11)
    assert np.allclose(nll, expected_nll, rtol=1e-12, atol=1e-12)
    assert np.allclose(grad, expected_grad[:-1].reshape(-1), rtol=1e-12, atol=1e-12)


def test_bootstrap_is_deterministic():
    rows = []
    for seq in range(4):
        for model, ll in [("Independent_switching", -10 - seq), ("HMM_static", -9 - seq)]:
            rows.append({"fold": 1, "seq_id": seq, "model": model, "test_ll": ll, "n_test": 10})
    df = pd.DataFrame(rows)
    a = bootstrap_model_differences(df, n_bootstrap=50, seed=7)
    b = bootstrap_model_differences(df, n_bootstrap=50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_descriptive_summaries_and_stationary_baseline(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    summary = dataset_summary(data).iloc[0]
    assert summary["raw_rows"] == 24
    assert summary["usable_trials"] == 23
    condition = behavioral_condition_summary(data)
    assert condition["n"].sum() == 23
    stationary = stationary_distribution(np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.2, 0.5]]))
    assert np.allclose(stationary.sum(), 1.0)
    assert np.all(stationary >= 0)


def test_posterior_occupancy_bins_and_parallel_restarts(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    one = fit_hmm_multistart(data, range(len(data.sequences)), n_restarts=2, seed0=11, max_iter=3, n_jobs=1)
    two = fit_hmm_multistart(data, range(len(data.sequences)), n_restarts=2, seed0=11, max_iter=3, n_jobs=2)
    assert one.seed == two.seed
    assert np.isclose(one.train_loglik, two.train_loglik)
    assert len(two.restart_diagnostics) == 2
    posterior = pd.DataFrame({
        "row_index": [0, 1, 2],
        "motion_direction": [225, 260, 330],
        "prior_mean": [225, 225, 225],
        "motion_coherence": [0.06, 0.12, 0.24],
        "prior_std": [10, 20, 40],
        "p_sensory": [0.8, 0.5, 0.2],
        "p_prior": [0.1, 0.4, 0.3],
        "p_lapse": [0.1, 0.1, 0.5],
    })
    occupancy = posterior_state_occupancy(posterior)
    assert set(occupancy["conflict_bin"]) == {"near (<30 deg)", "medium (30-60 deg)", "far (>=60 deg)"}


def test_covariate_effect_intervals_are_deterministic(tmp_path):
    csv = tmp_path / "small.csv"
    _write_small_csv(csv)
    data = load_direction_data(csv)
    params = init_covariate_hmm(data, seed=9)
    first = conditional_covariate_effect_intervals(data, params, n_bootstrap=8, seed=4)
    second = conditional_covariate_effect_intervals(data, params, n_bootstrap=8, seed=4)
    pd.testing.assert_frame_equal(first, second)
    assert {"ci_low", "ci_high", "p_two_sided", "method"}.issubset(first.columns)


def test_publication_gate_and_narrative_branch(tmp_path):
    pd.DataFrame({"converged": [False]}).to_csv(tmp_path / "cv_results.csv", index=False)
    pd.DataFrame({"converged": [False]}).to_csv(tmp_path / "model_info_criteria.csv", index=False)
    pd.DataFrame({"converged": [False]}).to_csv(tmp_path / "subject_level_hmm.csv", index=False)
    status, issues = publication_status(tmp_path, {})
    assert status == "partial"
    assert issues
    tables = {
        "cv_summary": pd.DataFrame({"model_label": ["Static HMM"], "mean_test_ll_per_trial": [-0.8]}),
        "bootstrap": pd.DataFrame({"model": ["HMM_static"], "observed_delta_ll_per_trial": [0.05], "ci_low": [0.02], "ci_high": [0.08]}),
        "subject": pd.DataFrame({"subject_id": [1, 1], "state": ["Sensory", "Prior"], "self_transition": [0.8, 0.85]}),
    }
    claims = result_claims(tables, status)
    assert "not for publication" in claims["status"]


def test_report_browser_style_is_idempotent(tmp_path):
    report = tmp_path / "report.html"
    report.write_text("<html><head><title>Test</title></head><body>Report</body></html>", encoding="utf-8")
    _inject_report_browser_style(report)
    _inject_report_browser_style(report)
    html = report.read_text(encoding="utf-8")
    assert html.count(_REPORT_BROWSER_STYLE_ID) == 1
    assert html.index(_REPORT_BROWSER_STYLE_ID) < html.index("</head>")


def test_plotting_smoke(tmp_path):
    cv_summary = pd.DataFrame({
        "model": ["HMM_static", "Independent_switching"],
        "mean_test_ll_per_trial": [-0.8, -0.9],
        "se_test_ll_per_trial": [0.01, 0.02],
    })
    cv_results = pd.DataFrame({
        "model": ["Serial_stim_independent_switching"],
        "fold": [1],
        "alpha_stim": [0.1],
        "alpha_resp": [0.0],
    })
    subject = pd.DataFrame({"subject_id": [1, 2], "A_SS": [0.8, 0.9], "A_PP": [0.85, 0.88]})
    effects = pd.DataFrame({
        "covariate": ["coherence"],
        "previous_state": ["S_sensory"],
        "next_state": ["S_sensory"],
        "delta_plus_minus": [0.1],
    })
    ppc = pd.DataFrame({
        "source": ["observed", "simulated"],
        "motion_coherence": [0.06, 0.06],
        "prior_std": [10, 10],
        "mean_abs_error_deg": [20, 22],
    })
    run_summary = pd.DataFrame({
        "source": ["posterior_map", "simulated"],
        "state": ["S_sensory", "S_sensory"],
        "mean_run_length": [5, 4],
    })
    made = generate_figures(
        tmp_path,
        cv_summary=cv_summary,
        cv_results=cv_results,
        transition_matrix=np.eye(3),
        subject_df=subject,
        covariate_effects=effects,
        ppc_summary=ppc,
        run_summary=run_summary,
    )
    assert made
    assert all(p.exists() and p.stat().st_size > 0 for p in made)
