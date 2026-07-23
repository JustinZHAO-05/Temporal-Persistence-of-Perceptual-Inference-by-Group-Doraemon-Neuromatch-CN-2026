"""Full-data AIC/BIC audit for the three subject-specific latent-state models."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
WORKSPACE = ROOT.parents[1]
ARTIFACTS = WORKSPACE / "hybrid" / "common"
if str(ARTIFACTS) not in sys.path:
    sys.path.insert(0, str(ARTIFACTS))

from markov_test_core import ensure, fit_hybrid, load_versions, modules


PARAMETERS_PER_SUBJECT = {
    "original_hmm": 15,  # pi:2 + A:6 + emissions:7
    "exchangeable_block_mixture": 15,  # rho:2 + class weights:6 + emissions:7
    "block_plus_markov": 33,  # rho:2 + class pi:6 + class A:18 + emissions:7
}


def _read_or_fit_hybrid(subject_id: int, data: pd.DataFrame, base, cache: Path) -> dict:
    result_path = cache / f"subject_{subject_id:02d}_result.json"
    history_path = cache / f"subject_{subject_id:02d}_history.csv"
    if result_path.exists() and history_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    fit = fit_hybrid(data.reset_index(drop=True), base)
    result = {
        "subject_id": subject_id,
        "n_trials": int(len(data)),
        "converged": bool(fit["converged"]),
        "n_iterations": int(fit["n_iterations"]),
        "final_log_likelihood": float(fit["history"][-1]),
        "monotonic": bool(fit["monotonic"]),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        {"iteration": np.arange(1, len(fit["history"]) + 1), "log_likelihood": fit["history"]}
    ).to_csv(history_path, index=False)
    return result


def _information_row(model_name: str, subject_id: int, n: int, ll: float, converged: bool) -> dict:
    k = PARAMETERS_PER_SUBJECT[model_name]
    return {
        "model_name": model_name,
        "subject_id": int(subject_id),
        "n_trials": int(n),
        "log_likelihood": float(ll),
        "parameters_per_subject": k,
        "aic": -2 * ll + 2 * k,
        "bic_subject": -2 * ll + k * math.log(n),
        "converged": bool(converged),
    }


def run_information_criteria() -> tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure(ROOT / "results")
    cache = ensure(out / "parameter_exports")
    base, _ = modules()
    original, _ = load_versions()

    hmm = pd.read_csv(WORKSPACE / "HMM" / "01_full_data_fit" / "results" / "all_subject_summary_revised.csv")
    hmm = hmm.rename(columns={"valid_n_trials": "n_trials", "final_log_likelihood": "log_likelihood"})
    block = pd.read_csv(WORKSPACE / "exchangeable_block_mixture" / "results" / "heldout" / "full_fit_summary.csv")
    block = block[block["model_name"] == "exchangeable_block_mixture"].copy()

    rows = []
    for row in hmm.itertuples(index=False):
        rows.append(
            _information_row(
                "original_hmm", row.subject_id, row.n_trials, row.log_likelihood, row.converged
            )
        )
    for row in block.itertuples(index=False):
        rows.append(
            _information_row(
                "exchangeable_block_mixture",
                row.subject_id,
                row.n_trials,
                row.final_log_likelihood,
                row.converged,
            )
        )

    hybrid_rows = []
    for subject_id, data in original.groupby("subject_id", sort=True):
        result = _read_or_fit_hybrid(int(subject_id), data, base, cache)
        hybrid_rows.append(result)
        print(f"hybrid full fit subject {subject_id}: {result['n_iterations']} iterations")
        rows.append(
            _information_row(
                "block_plus_markov",
                result["subject_id"],
                result["n_trials"],
                result["final_log_likelihood"],
                result["converged"],
            )
        )

    subject = pd.DataFrame(rows).sort_values(["model_name", "subject_id"])
    subject.to_csv(out / "subject_information_criteria.csv", index=False)
    pd.DataFrame(hybrid_rows).to_csv(out / "hybrid_full_fit_summary.csv", index=False)

    overall = (
        subject.groupby("model_name", as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_trials=("n_trials", "sum"),
            total_log_likelihood=("log_likelihood", "sum"),
            total_parameters=("parameters_per_subject", "sum"),
            aic_sum=("aic", "sum"),
            bic_sum_subjectwise=("bic_subject", "sum"),
            converged_subjects=("converged", "sum"),
        )
    )
    overall["bic_pooled_trial_convention"] = (
        -2 * overall["total_log_likelihood"]
        + overall["total_parameters"] * np.log(overall["n_trials"])
    )
    overall["aic_delta_from_best"] = overall["aic_sum"] - overall["aic_sum"].min()
    overall["bic_subjectwise_delta_from_best"] = (
        overall["bic_sum_subjectwise"] - overall["bic_sum_subjectwise"].min()
    )
    overall["bic_pooled_delta_from_best"] = (
        overall["bic_pooled_trial_convention"] - overall["bic_pooled_trial_convention"].min()
    )
    overall = overall.sort_values("aic_sum")
    overall.to_csv(out / "overall_information_criteria.csv", index=False)

    validation = {
        "three_models": int(overall.shape[0]) == 3,
        "twelve_subjects_each": bool((overall["n_subjects"] == 12).all()),
        "finite": bool(np.isfinite(subject[["log_likelihood", "aic", "bic_subject"]]).all().all()),
        "parameter_counts_expected": bool(
            set(overall["total_parameters"]) == {180, 396}
        ),
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not all(validation.values()):
        raise AssertionError(validation)
    return subject, overall
