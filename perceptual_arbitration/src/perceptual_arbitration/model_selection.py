from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

from .data import DataBundle, sequence_labels_for_cv, transition_scaler_from_sequences, with_transition_scaler
from .diagnostics import (
    covariate_coefficient_table,
    covariate_effect_table,
    information_criteria,
)
from .hmm import fit_hmm_multistart, loglik_hmm, posterior_hmm, STATE_NAMES
from .independent_switching import fit_independent_multistart, loglik_independent
from .serial_dependence import fit_serial_baseline, loglik_serial
from .covariate_hmm import fit_covariate_hmm_multistart, loglik_covariate_hmm, average_transition_matrix
from .checkpoints import load_fit_checkpoint, save_fit_checkpoint


def _fit_or_resume(
    out_dir: Path,
    checkpoint_name: str,
    run_key: str,
    resume: bool,
    fit_fn: Callable,
):
    path = out_dir / "checkpoints" / f"{checkpoint_name}.joblib"
    if resume:
        fit = load_fit_checkpoint(path, run_key)
        if fit is not None:
            return fit
    fit = fit_fn()
    save_fit_checkpoint(path, run_key, fit)
    return fit


def _restart_rows(fit, stage: str, model: str, fold: int | None = None, subject_id=None) -> list[dict]:
    rows = []
    for row in fit.restart_diagnostics or []:
        rows.append({
            "stage": stage,
            "fold": fold,
            "subject_id": subject_id,
            "model": model,
            **row,
        })
    return rows


def _stationary_distribution(A: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(np.asarray(A, dtype=float).T)
    raw = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    if raw.sum() < 0:
        raw = -raw
    vector = np.maximum(raw, 0.0)
    if vector.sum() <= 0:
        vector = np.abs(raw)
    if vector.sum() <= 0:
        vector = np.ones(len(raw), dtype=float)
    return vector / vector.sum()


def n_trials_for_sequences(data: DataBundle, seq_ids) -> int:
    return int(data.seq_meta.loc[list(seq_ids), "n"].sum())


def make_cv_splits(data: DataBundle, n_splits: int = 4, seed: int = 42, group_by_subject: bool = False):
    seq_ids = np.arange(len(data.sequences))
    if group_by_subject:
        # Leave-subject/generalization split. Requires n_splits <= number of subjects.
        groups = data.seq_meta["subject_id"].to_numpy()
        gkf = GroupKFold(n_splits=n_splits)
        labels = sequence_labels_for_cv(data)
        yield from ((seq_ids[tr], seq_ids[te]) for tr, te in gkf.split(seq_ids, labels, groups=groups))
    else:
        labels = sequence_labels_for_cv(data)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from ((seq_ids[tr], seq_ids[te]) for tr, te in skf.split(seq_ids, labels))


def _append_per_sequence_rows(rows: list[dict], data: DataBundle, fold: int, model: str, params, seq_ids, loglik_fn) -> None:
    for sid in seq_ids:
        seq = np.array([int(sid)])
        ll = float(loglik_fn(data, params, seq))
        n = int(data.seq_meta.loc[int(sid), "n"])
        meta = data.seq_meta.loc[int(sid)]
        rows.append({
            "fold": fold,
            "model": model,
            "seq_id": int(sid),
            "subject_id": meta["subject_id"],
            "session_id": meta["session_id"],
            "run_id": meta["run_id"],
            "prior_std": meta["prior_std"],
            "n_test": n,
            "test_ll": ll,
            "test_ll_per_trial": ll / n,
        })


def run_core_cv(
    data: DataBundle,
    out_dir: str | Path,
    n_splits: int = 4,
    n_restarts_hmm: int = 25,
    n_restarts_ind: int = 25,
    n_restarts_serial: int = 10,
    n_restarts_iohmm: int = 10,
    max_iter_em: int = 1000,
    seed: int = 42,
    include_iohmm: bool = True,
    include_serial: bool = True,
    tol_hmm: float = 1e-7,
    n_jobs: int = 1,
    resume: bool = False,
    run_key: str = "unversioned",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_sequence_rows = []
    restart_rows = []
    covariate_fold_effects = []

    def flush_live() -> None:
        pd.DataFrame(rows).to_csv(out_dir / "cv_results_live.csv", index=False)
        pd.DataFrame(per_sequence_rows).to_csv(out_dir / "per_sequence_cv_results_live.csv", index=False)
        pd.DataFrame(restart_rows).to_csv(out_dir / "restart_diagnostics_cv_live.csv", index=False)

    for fold, (train_seq, test_seq) in enumerate(make_cv_splits(data, n_splits=n_splits, seed=seed), start=1):
        ntrain = n_trials_for_sequences(data, train_seq)
        ntest = n_trials_for_sequences(data, test_seq)

        hmm = _fit_or_resume(
            out_dir,
            f"cv_fold_{fold}_hmm_static",
            run_key,
            resume,
            lambda: fit_hmm_multistart(
                data,
                train_seq,
                n_restarts=n_restarts_hmm,
                max_iter=max_iter_em,
                tol=tol_hmm,
                seed0=1000 + 100 * fold,
                n_jobs=n_jobs,
            ),
        )
        hmm_test = loglik_hmm(data, hmm.params, test_seq)
        _append_per_sequence_rows(per_sequence_rows, data, fold, "HMM_static", hmm.params, test_seq, loglik_hmm)
        rows.append({
            "fold": fold, "model": "HMM_static", "n_train": ntrain, "n_test": ntest,
            "train_ll": hmm.train_loglik, "test_ll": hmm_test, "test_ll_per_trial": hmm_test / ntest,
            "converged": hmm.converged, "n_iter": hmm.n_iter, "seed": hmm.seed,
            "A_SS": hmm.params.A[0, 0], "A_PP": hmm.params.A[1, 1], "A_LL": hmm.params.A[2, 2],
        })
        restart_rows.extend(_restart_rows(hmm, "cv", "HMM_static", fold=fold))
        flush_live()

        ind = _fit_or_resume(
            out_dir,
            f"cv_fold_{fold}_independent_switching",
            run_key,
            resume,
            lambda: fit_independent_multistart(
                data,
                train_seq,
                n_restarts=n_restarts_ind,
                max_iter=max_iter_em,
                seed0=2000 + 100 * fold,
                n_jobs=n_jobs,
            ),
        )
        ind_test = loglik_independent(data, ind.params, test_seq)
        _append_per_sequence_rows(per_sequence_rows, data, fold, "Independent_switching", ind.params, test_seq, loglik_independent)
        rows.append({
            "fold": fold, "model": "Independent_switching", "n_train": ntrain, "n_test": ntest,
            "train_ll": ind.train_loglik, "test_ll": ind_test, "test_ll_per_trial": ind_test / ntest,
            "converged": ind.converged, "n_iter": ind.n_iter, "seed": ind.seed,
        })
        restart_rows.extend(_restart_rows(ind, "cv", "Independent_switching", fold=fold))
        flush_live()

        if include_serial:
            for mode in ["stim", "resp", "both"]:
                model_name = f"Serial_{mode}_independent_switching"
                sd = _fit_or_resume(
                    out_dir,
                    f"cv_fold_{fold}_serial_{mode}",
                    run_key,
                    resume,
                    lambda mode=mode: fit_serial_baseline(
                        data,
                        train_seq,
                        mode=mode,
                        n_restarts=n_restarts_serial,
                        seed0=3000 + 100 * fold,
                        maxiter=max_iter_em,
                        n_jobs=n_jobs,
                    ),
                )
                sd_test = loglik_serial(data, sd.params, test_seq)
                _append_per_sequence_rows(per_sequence_rows, data, fold, model_name, sd.params, test_seq, loglik_serial)
                rows.append({
                    "fold": fold, "model": model_name, "n_train": ntrain, "n_test": ntest,
                    "train_ll": sd.train_loglik, "test_ll": sd_test, "test_ll_per_trial": sd_test / ntest,
                    "converged": sd.converged, "n_iter": sd.n_iter, "seed": sd.seed,
                    "alpha_stim": sd.params.alpha_stim, "alpha_resp": sd.params.alpha_resp,
                })
                restart_rows.extend(_restart_rows(sd, "cv", model_name, fold=fold))
                flush_live()

        if include_iohmm:
            means, sds = transition_scaler_from_sequences(data, train_seq)
            data_cov = with_transition_scaler(data, means, sds)
            iohmm = _fit_or_resume(
                out_dir,
                f"cv_fold_{fold}_covariate_hmm",
                run_key,
                resume,
                lambda: fit_covariate_hmm_multistart(
                    data_cov,
                    train_seq,
                    n_restarts=n_restarts_iohmm,
                    max_iter=max_iter_em,
                    seed0=4000 + 100 * fold,
                    n_jobs=n_jobs,
                ),
            )
            io_test = loglik_covariate_hmm(data_cov, iohmm.params, test_seq)
            avg_A = average_transition_matrix(data_cov, iohmm.params, test_seq)
            _append_per_sequence_rows(per_sequence_rows, data_cov, fold, "Covariate_HMM", iohmm.params, test_seq, loglik_covariate_hmm)
            rows.append({
                "fold": fold, "model": "Covariate_HMM", "n_train": ntrain, "n_test": ntest,
                "train_ll": iohmm.train_loglik, "test_ll": io_test, "test_ll_per_trial": io_test / ntest,
                "converged": iohmm.converged, "n_iter": iohmm.n_iter, "seed": iohmm.seed,
                "avg_A_SS": avg_A[0, 0], "avg_A_PP": avg_A[1, 1], "avg_A_LL": avg_A[2, 2],
            })
            effects = covariate_effect_table(iohmm.params, data_cov.transition_names)
            effects.insert(0, "fold", fold)
            covariate_fold_effects.extend(effects.to_dict("records"))
            restart_rows.extend(_restart_rows(iohmm, "cv", "Covariate_HMM", fold=fold))
            flush_live()

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "cv_results.csv", index=False)
    pd.DataFrame(per_sequence_rows).to_csv(out_dir / "per_sequence_cv_results.csv", index=False)
    pd.DataFrame(restart_rows).to_csv(out_dir / "restart_diagnostics_cv.csv", index=False)
    pd.DataFrame(covariate_fold_effects).to_csv(out_dir / "covariate_hmm_fold_effects.csv", index=False)
    return results


def summarize_cv(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("model").agg(
        folds=("fold", "nunique"),
        mean_test_ll_per_trial=("test_ll_per_trial", "mean"),
        se_test_ll_per_trial=("test_ll_per_trial", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_test_ll=("test_ll", "mean"),
    ).reset_index()
    best = summary["mean_test_ll_per_trial"].max()
    summary["delta_from_best_per_trial"] = summary["mean_test_ll_per_trial"] - best
    return summary.sort_values("mean_test_ll_per_trial", ascending=False)


def fit_subject_level_hmms(
    data: DataBundle,
    out_dir: str | Path,
    n_restarts: int = 25,
    max_iter: int = 1000,
    tol: float = 1e-7,
    n_jobs: int = 1,
    resume: bool = False,
    run_key: str = "unversioned",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    restart_rows = []
    for subj in data.subject_values:
        seq_ids = data.seq_meta.index[data.seq_meta["subject_id"] == subj].to_numpy()
        fit = _fit_or_resume(
            out_dir,
            f"subject_{int(subj)}_hmm",
            run_key,
            resume,
            lambda: fit_hmm_multistart(
                data,
                seq_ids,
                n_restarts=n_restarts,
                max_iter=max_iter,
                tol=tol,
                seed0=7000 + int(subj) * 100,
                n_jobs=n_jobs,
            ),
        )
        stationary = _stationary_distribution(fit.params.A)
        rows.append({
            "subject_id": subj,
            "n_sequences": len(seq_ids),
            "n_trials": n_trials_for_sequences(data, seq_ids),
            "train_ll": fit.train_loglik,
            "ll_per_trial": fit.train_loglik / n_trials_for_sequences(data, seq_ids),
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "A_SS": fit.params.A[0, 0],
            "A_SP": fit.params.A[0, 1],
            "A_SL": fit.params.A[0, 2],
            "A_PS": fit.params.A[1, 0],
            "A_PP": fit.params.A[1, 1],
            "A_PL": fit.params.A[1, 2],
            "A_LS": fit.params.A[2, 0],
            "A_LP": fit.params.A[2, 1],
            "A_LL": fit.params.A[2, 2],
            "stationary_S": stationary[0],
            "stationary_P": stationary[1],
            "stationary_L": stationary[2],
            "persistence_excess_S": fit.params.A[0, 0] - stationary[0],
            "persistence_excess_P": fit.params.A[1, 1] - stationary[1],
            **{f"kappaS_{c}": k for c, k in zip(data.coh_values, fit.params.kappa_s)},
            **{f"kappaP_{p}": k for p, k in zip(data.prior_values, fit.params.kappa_p)},
        })
        restart_rows.extend(_restart_rows(fit, "subject", "HMM_static", subject_id=subj))
        pd.DataFrame(rows).to_csv(out_dir / "subject_level_hmm_live.csv", index=False)
        pd.DataFrame(restart_rows).to_csv(out_dir / "restart_diagnostics_subject_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "subject_level_hmm.csv", index=False)
    pd.DataFrame(restart_rows).to_csv(out_dir / "restart_diagnostics_subject.csv", index=False)
    return df


def empirical_bayes_summary(subject_df: pd.DataFrame, out_dir: str | Path):
    """Lightweight hierarchical summary: group mean and between-subject SD.

    For a full paper, replace or supplement this with Bayesian random-effects
    inference in Stan/PyMC/NumPyro. This summary is still useful as an empirical
    Bayes sanity check.
    """
    numeric = subject_df.select_dtypes(include=[np.number]).drop(columns=["subject_id"], errors="ignore")
    summary = pd.DataFrame({
        "parameter": numeric.columns,
        "mean": numeric.mean().values,
        "sd_between_subjects": numeric.std(ddof=1).values,
        "se": numeric.std(ddof=1).values / np.sqrt(len(numeric)),
    })
    out_dir = Path(out_dir)
    summary.to_csv(out_dir / "empirical_bayes_group_summary.csv", index=False)
    return summary


def fit_final_models(
    data: DataBundle,
    out_dir: str | Path,
    n_restarts_hmm: int = 25,
    n_restarts_ind: int = 25,
    n_restarts_serial: int = 15,
    n_restarts_iohmm: int = 10,
    max_iter_em: int = 1000,
    include_iohmm: bool = True,
    include_serial: bool = True,
    tol_hmm: float = 1e-7,
    n_jobs: int = 1,
    resume: bool = False,
    run_key: str = "unversioned",
):
    """Fit all-data models and write publication-grade parameter diagnostics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_ids = np.arange(len(data.sequences))
    info_rows: list[dict] = []
    restart_rows: list[dict] = []

    hmm = _fit_or_resume(
        out_dir,
        "final_hmm_static",
        run_key,
        resume,
        lambda: fit_hmm_multistart(
            data,
            seq_ids,
            n_restarts=n_restarts_hmm,
            max_iter=max_iter_em,
            tol=tol_hmm,
            seed0=9000,
            n_jobs=n_jobs,
        ),
    )
    restart_rows.extend(_restart_rows(hmm, "final", "HMM_static"))
    pd.DataFrame(hmm.params.A, index=STATE_NAMES, columns=STATE_NAMES).to_csv(out_dir / "hmm_final_transition_matrix.csv")
    param_rows = []
    for c, k in zip(data.coh_values, hmm.params.kappa_s):
        param_rows.append({"parameter": "sensory_kappa", "condition": f"coherence_{c:g}", "value": k})
    for p, k in zip(data.prior_values, hmm.params.kappa_p):
        param_rows.append({"parameter": "prior_kappa", "condition": f"prior_std_{p:g}", "value": k})
    for state, prob in zip(STATE_NAMES, hmm.params.pi):
        param_rows.append({"parameter": "initial_probability", "condition": state, "value": prob})
    param_rows.append({"parameter": "full_data_loglik", "condition": "HMM_static", "value": hmm.train_loglik})
    pd.DataFrame(param_rows).to_csv(out_dir / "hmm_final_parameters.csv", index=False)
    posterior = posterior_hmm(data, hmm.params, seq_ids)
    posterior.to_csv(out_dir / "posterior_states.csv", index=False)
    info_rows.append({
        "model": "HMM_static",
        "train_ll": hmm.train_loglik,
        "converged": hmm.converged,
        "n_iter": hmm.n_iter,
        "seed": hmm.seed,
    })

    ind = _fit_or_resume(
        out_dir,
        "final_independent_switching",
        run_key,
        resume,
        lambda: fit_independent_multistart(
            data,
            seq_ids,
            n_restarts=n_restarts_ind,
            max_iter=max_iter_em,
            seed0=10000,
            n_jobs=n_jobs,
        ),
    )
    restart_rows.extend(_restart_rows(ind, "final", "Independent_switching"))
    info_rows.append({
        "model": "Independent_switching",
        "train_ll": ind.train_loglik,
        "converged": ind.converged,
        "n_iter": ind.n_iter,
        "seed": ind.seed,
    })

    serial_fits = {}
    if include_serial:
        for mode in ["stim", "resp", "both"]:
            model_name = f"Serial_{mode}_independent_switching"
            fit = _fit_or_resume(
                out_dir,
                f"final_serial_{mode}",
                run_key,
                resume,
                lambda mode=mode: fit_serial_baseline(
                    data,
                    seq_ids,
                    mode=mode,
                    n_restarts=n_restarts_serial,
                    seed0=11000,
                    maxiter=max_iter_em,
                    n_jobs=n_jobs,
                ),
            )
            serial_fits[mode] = fit
            restart_rows.extend(_restart_rows(fit, "final", model_name))
            info_rows.append({
                "model": model_name,
                "train_ll": fit.train_loglik,
                "converged": fit.converged,
                "n_iter": fit.n_iter,
                "seed": fit.seed,
                "alpha_stim": fit.params.alpha_stim,
                "alpha_resp": fit.params.alpha_resp,
            })

    covariate = None
    covariate_coefficients = pd.DataFrame()
    covariate_effects = pd.DataFrame()
    if include_iohmm:
        covariate = _fit_or_resume(
            out_dir,
            "final_covariate_hmm",
            run_key,
            resume,
            lambda: fit_covariate_hmm_multistart(
                data,
                seq_ids,
                n_restarts=n_restarts_iohmm,
                max_iter=max_iter_em,
                seed0=12000,
                n_jobs=n_jobs,
            ),
        )
        restart_rows.extend(_restart_rows(covariate, "final", "Covariate_HMM"))
        avg_A = average_transition_matrix(data, covariate.params, seq_ids)
        pd.DataFrame(avg_A, index=STATE_NAMES, columns=STATE_NAMES).to_csv(out_dir / "covariate_hmm_average_transition_matrix.csv")
        covariate_coefficients = covariate_coefficient_table(covariate.params, data.transition_names)
        covariate_effects = covariate_effect_table(covariate.params, data.transition_names)
        covariate_coefficients.to_csv(out_dir / "covariate_hmm_coefficients.csv", index=False)
        covariate_effects.to_csv(out_dir / "covariate_hmm_effects.csv", index=False)
        info_rows.append({
            "model": "Covariate_HMM",
            "train_ll": covariate.train_loglik,
            "converged": covariate.converged,
            "n_iter": covariate.n_iter,
            "seed": covariate.seed,
            "avg_A_SS": avg_A[0, 0],
            "avg_A_PP": avg_A[1, 1],
            "avg_A_LL": avg_A[2, 2],
        })

    info = information_criteria(info_rows, data)
    info.to_csv(out_dir / "model_info_criteria.csv", index=False)
    pd.DataFrame(restart_rows).to_csv(out_dir / "restart_diagnostics_final.csv", index=False)
    return {
        "hmm": hmm,
        "independent": ind,
        "serial": serial_fits,
        "covariate": covariate,
        "posterior": posterior,
        "model_info": info,
        "covariate_coefficients": covariate_coefficients,
        "covariate_effects": covariate_effects,
    }
