"""Read-only consistency checks for the curated contribution bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def require(path: str) -> Path:
    candidate = ROOT / path
    if not candidate.exists():
        raise AssertionError(f"Missing required path: {path}")
    return candidate


def main() -> dict[str, bool]:
    checks: dict[str, bool] = {}

    hmm = pd.read_csv(
        require("HMM/01_full_data_fit/results/all_subject_summary_revised.csv")
    )
    checks["hmm_has_12_subjects"] = len(hmm) == 12 and hmm["subject_id"].nunique() == 12
    checks["hmm_has_83210_valid_trials"] = int(hmm["valid_n_trials"].sum()) == 83210

    hmm_cv = pd.read_csv(
        require("HMM/02_fourfold_cv/results/our_model_fourfold_overall_summary.csv")
    ).iloc[0]
    checks["hmm_cv_has_83210_trials"] = int(hmm_cv["n_trials"]) == 83210
    checks["hmm_cv_ll_matches_report"] = np.isclose(
        float(hmm_cv["mean_held_out_ll_per_trial"]), -0.810568, atol=1e-6
    )

    original = pd.read_csv(
        require(
            "HMM/03_shuffle_control/results/heldout/"
            "all_subject_4fold_predictions_original.csv"
        )
    )
    shuffled = pd.read_csv(
        require(
            "HMM/03_shuffle_control/results/heldout/"
            "all_subject_4fold_predictions_shuffled.csv"
        )
    )
    checks["shuffle_versions_each_have_83210_trials"] = (
        len(original) == len(shuffled) == 83210
    )
    checks["shuffle_uses_same_trial_set"] = set(original["original_row_index"]) == set(
        shuffled["original_row_index"]
    )

    block = pd.read_csv(
        require(
            "exchangeable_block_mixture/results/comparisons/"
            "overall_model_comparison.csv"
        )
    )
    block_row = block[
        (block["model_name"] == "exchangeable_block_mixture")
        & (block["data_version"] == "original")
    ].iloc[0]
    checks["block_mixture_ll_matches_report"] = np.isclose(
        float(block_row["ll_per_trial"]), -0.794568, atol=1e-6
    )

    hybrid = pd.read_csv(
        require("hybrid/01_random_fourfold/results/comparison.csv")
    ).set_index("model")
    checks["hybrid_ll_matches_report"] = np.isclose(
        float(hybrid.loc["block_plus_markov", "ll_per_trial"]), -0.787089, atol=1e-6
    )

    criteria = pd.read_csv(
        require(
            "hybrid/03_information_criteria/results/"
            "overall_information_criteria.csv"
        )
    )
    checks["information_criteria_has_three_models"] = (
        len(criteria) == 3 and criteria["model_name"].nunique() == 3
    )
    checks["aic_selects_hybrid"] = (
        criteria.sort_values("aic_sum").iloc[0]["model_name"] == "block_plus_markov"
    )
    checks["bic_selects_block_mixture"] = (
        criteria.sort_values("bic_sum_subjectwise").iloc[0]["model_name"]
        == "exchangeable_block_mixture"
    )

    viterbi = pd.read_csv(
        require("HMM/04_viterbi_trajectory/results/all_subjects_viterbi_states.csv")
    )
    checks["viterbi_has_83210_trials"] = len(viterbi) == 83210

    reproduced_viterbi = pd.read_csv(
        require(
            "HMM/04_viterbi_trajectory/results/reproduced/"
            "all_subjects_viterbi_states.csv"
        )
    )
    checks["viterbi_rerun_matches_saved_output"] = viterbi.equals(reproduced_viterbi)

    full_parameters = pd.read_csv(
        require(
            "hybrid/03_information_criteria/results/parameter_exports/"
            "parameter_export_summary.csv"
        )
    )
    checks["hybrid_full_parameters_reproduced_12_of_12"] = (
        len(full_parameters) == 12
        and full_parameters["matches_reference_within_1e-8"].all()
    )

    random_parameters = pd.read_csv(
        require(
            "hybrid/01_random_fourfold/results/parameter_exports/"
            "parameter_export_summary.csv"
        )
    )
    checks["hybrid_random_parameters_reproduced_48_of_48"] = (
        len(random_parameters) == 48
        and random_parameters["matches_reference_within_1e-8"].all()
    )

    chronological_parameters = pd.read_csv(
        require(
            "hybrid/02_chronological_prediction/results/parameter_exports/"
            "parameter_export_summary.csv"
        )
    )
    checks["hybrid_chronological_parameters_reproduced_48_of_48"] = (
        len(chronological_parameters) == 48
        and chronological_parameters["matches_reference_within_1e-8"].all()
    )

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Bundle validation failed: {failed}")
    checks = {name: bool(value) for name, value in checks.items()}
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    main()
