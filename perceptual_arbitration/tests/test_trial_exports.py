from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from perceptual_arbitration.checkpoints import save_fit_checkpoint
from perceptual_arbitration.covariate_hmm import (
    CovariateHMMFit,
    init_covariate_hmm,
    loglik_covariate_hmm,
)
from perceptual_arbitration.data import load_direction_data
from perceptual_arbitration.hmm import HMMFit, HMMParams, loglik_hmm
from perceptual_arbitration.independent_switching import IndepFit, IndepParams, loglik_independent
from perceptual_arbitration.run_metadata import atomic_write_json, sha256_file
from perceptual_arbitration.serial_dependence import SerialFit, SerialParams, loglik_serial
from perceptual_arbitration.trial_exports import (
    TRANSITION_COLUMNS,
    XI_COLUMNS,
    build_trial_base,
    export_trial_results,
    score_hmm,
    score_independent_or_serial,
)


def _xy(angle_deg: float) -> tuple[float, float]:
    angle = np.deg2rad(angle_deg)
    return float(np.cos(angle)), float(np.sin(angle))


def _write_csv(path: Path) -> None:
    rows = []
    for subject in [1, 2]:
        for run in [1, 2]:
            for trial in range(1, 7):
                direction = 210 + 10 * trial
                estimate = direction + (-5 if trial % 2 else 8)
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
                    "prior_std": [10, 20][run - 1],
                    "prior_mean": 225,
                    "subject_id": subject,
                    "experiment_name": "synthetic",
                    "experiment_id": 99,
                    "session_id": 1,
                    "run_id": run,
                })
    rows[4]["estimate_x"] = np.nan
    rows[4]["estimate_y"] = np.nan
    pd.DataFrame(rows).to_csv(path, index=False)


def _parameters(data):
    kappa_s = np.array([2.0, 5.0, 12.0])
    kappa_p = np.array([18.0, 4.0])
    independent = IndepParams(w=np.array([0.55, 0.35, 0.10]), kappa_s=kappa_s, kappa_p=kappa_p)
    serial = {
        "stim": SerialParams(np.array([0.55, 0.35, 0.10]), kappa_s, kappa_p, 0.08, 0.0),
        "resp": SerialParams(np.array([0.55, 0.35, 0.10]), kappa_s, kappa_p, 0.0, -0.05),
        "both": SerialParams(np.array([0.55, 0.35, 0.10]), kappa_s, kappa_p, 0.08, -0.05),
    }
    hmm = HMMParams(
        pi=np.array([0.6, 0.3, 0.1]),
        A=np.array([[0.8, 0.15, 0.05], [0.12, 0.82, 0.06], [0.4, 0.2, 0.4]]),
        kappa_s=kappa_s,
        kappa_p=kappa_p,
    )
    covariate = init_covariate_hmm(data, seed=7)
    covariate.kappa_s = kappa_s.copy()
    covariate.kappa_p = kappa_p.copy()
    return independent, serial, hmm, covariate


def test_mixture_and_serial_trial_scores(tmp_path):
    csv_path = tmp_path / "trials.csv"
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    independent, serial, _, _ = _parameters(data)
    seq_ids = np.arange(len(data.sequences))

    score = score_independent_or_serial(data, independent, seq_ids, serial=False)
    assert np.isclose(score["trial_log_predictive_density"].sum(), loglik_independent(data, independent, seq_ids))
    assert np.allclose(score[["filtered_p_sensory", "filtered_p_prior", "filtered_p_lapse"]].sum(axis=1), 1.0)
    assert np.array_equal(score["sensory_kappa"].to_numpy(), independent.kappa_s[data.coh_idx])
    assert np.array_equal(score["prior_kappa"].to_numpy(), independent.kappa_p[data.prior_idx])

    serial_score = score_independent_or_serial(data, serial["both"], seq_ids, serial=True)
    starts = np.concatenate([idx[:1] for idx in data.sequences])
    assert serial_score.loc[starts, "serial_sequence_start_current_trial_fallback"].all()
    expected_start_shift = serial["both"].alpha_resp * np.rad2deg(
        np.angle(np.exp(1j * (data.y[starts] - data.theta[starts])))
    )
    assert np.allclose(
        serial_score.loc[starts, "serial_sensory_center_shift_deg"],
        expected_start_shift,
    )
    assert np.isclose(serial_score["trial_log_predictive_density"].sum(), loglik_serial(data, serial["both"], seq_ids))
    base = build_trial_base(data)
    assert base.loc[base["is_sequence_start"], [column for column in base if column.startswith("previous_")]].isna().all().all()


def test_static_and_covariate_hmm_trial_scores(tmp_path):
    csv_path = tmp_path / "trials.csv"
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    _, _, hmm, covariate = _parameters(data)
    seq_ids = np.arange(len(data.sequences))

    static_score = score_hmm(data, hmm, seq_ids, covariate=False)
    assert np.isclose(static_score["trial_log_predictive_density"].sum(), loglik_hmm(data, hmm, seq_ids), atol=1e-10)
    starts = np.concatenate([idx[:1] for idx in data.sequences])
    nonstarts = static_score.index.difference(starts)
    assert static_score.loc[starts, [*TRANSITION_COLUMNS, *XI_COLUMNS]].isna().all().all()
    assert np.allclose(static_score.loc[nonstarts, list(XI_COLUMNS)].sum(axis=1), 1.0)
    assert np.allclose(static_score[["smoothed_p_sensory", "smoothed_p_prior", "smoothed_p_lapse"]].sum(axis=1), 1.0)

    cov_score = score_hmm(data, covariate, seq_ids, covariate=True)
    assert np.isclose(
        cov_score["trial_log_predictive_density"].sum(),
        loglik_covariate_hmm(data, covariate, seq_ids),
        atol=1e-10,
    )
    for source in ["sensory", "prior", "lapse"]:
        columns = [f"transition_p_{source}_to_{target}" for target in ["sensory", "prior", "lapse"]]
        assert np.allclose(cov_score.loc[nonstarts, columns].sum(axis=1), 1.0)
    assert "transition_covariate_coherence_raw" in cov_score
    assert "transition_scaler_mean_coherence" in cov_score


def test_final_export_from_checkpoints_is_complete_and_deterministic(tmp_path):
    csv_path = tmp_path / "trials.csv"
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()
    _write_csv(csv_path)
    data = load_direction_data(csv_path)
    independent, serial, hmm, covariate = _parameters(data)
    seq_ids = np.arange(len(data.sequences))
    run_key = "synthetic-run-key"

    fits = {
        "final_independent_switching": IndepFit(
            independent, loglik_independent(data, independent, seq_ids), 1, True, 1, []
        ),
        "final_serial_stim": SerialFit(
            serial["stim"], loglik_serial(data, serial["stim"], seq_ids), True, 2, "ok", 1
        ),
        "final_serial_resp": SerialFit(
            serial["resp"], loglik_serial(data, serial["resp"], seq_ids), True, 3, "ok", 1
        ),
        "final_serial_both": SerialFit(
            serial["both"], loglik_serial(data, serial["both"], seq_ids), True, 4, "ok", 1
        ),
        "final_hmm_static": HMMFit(hmm, loglik_hmm(data, hmm, seq_ids), 1, True, 5, []),
        "final_covariate_hmm": CovariateHMMFit(
            covariate,
            loglik_covariate_hmm(data, covariate, seq_ids),
            1,
            True,
            6,
            [],
        ),
    }
    for name, fit in fits.items():
        save_fit_checkpoint(out_dir / "checkpoints" / f"{name}.joblib", run_key, fit)

    model_names = [
        "Independent_switching",
        "Serial_stim_independent_switching",
        "Serial_resp_independent_switching",
        "Serial_both_independent_switching",
        "HMM_static",
        "Covariate_HMM",
    ]
    pd.DataFrame({
        "model": model_names,
        "train_ll": [fit.train_loglik for fit in fits.values()],
    }).to_csv(out_dir / "model_info_criteria.csv", index=False)
    parent_manifest = {
        "status": "complete",
        "completed_at": "2026-01-01T00:00:00+00:00",
        "publication_ready": True,
        "run_key": run_key,
        "config_hash": "config",
        "data_sha256": sha256_file(csv_path),
        "data": {"usable_trials": len(data.df)},
        "resolved_config": {"cv": {"n_splits": 4}, "diagnostics": {"seed": 42}},
        "stages": {
            "data": {"status": "complete"},
            "cross_validation": {"status": "complete"},
            "final_models": {"status": "complete"},
        },
    }
    atomic_write_json(parent_manifest, out_dir / "run_manifest.json")

    first = export_trial_results(
        csv_path,
        out_dir,
        scope="final",
        include_subject=False,
        overwrite=False,
    )
    assert first["usable_trials"] == 23
    assert first["excluded_trials"] == 1
    detailed_path = out_dir / "trial_exports" / "final" / "static_hmm_trials.csv"
    detailed = pd.read_csv(detailed_path)
    assert len(detailed) == 23
    assert detailed["analysis_row_index"].nunique() == 23
    first_hash = sha256_file(detailed_path)

    second = export_trial_results(
        csv_path,
        out_dir,
        scope="final",
        include_subject=False,
        overwrite=True,
    )
    assert second["status"] == "complete"
    assert sha256_file(detailed_path) == first_hash
    exported_manifest = json.loads((out_dir / "trial_exports" / "export_manifest.json").read_text())
    assert exported_manifest["fit_independent"] is True
    assert len(exported_manifest["unavailable_contextual_models"]) == 2
