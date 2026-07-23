"""Create clearly named, model-specific submission tables from migrated results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def write_exchangeable_block_tables() -> None:
    base = ROOT / "exchangeable_block_mixture" / "results"
    source = base / "heldout"

    full = pd.read_csv(source / "full_fit_summary.csv")
    full = full[full["model_name"] == "exchangeable_block_mixture"].copy()
    if len(full) != 12:
        raise AssertionError(f"Expected 12 block-mixture full fits, found {len(full)}")
    full.to_csv(
        base / "full_fit" / "exchangeable_block_mixture_full_fit_summary.csv",
        index=False,
    )

    folds = pd.read_csv(source / "fold_level_results.csv")
    folds = folds[folds["model_name"] == "exchangeable_block_mixture"].copy()
    if len(folds) != 96:
        raise AssertionError(f"Expected 96 block-mixture fold rows, found {len(folds)}")
    folds.to_csv(
        source / "exchangeable_block_mixture_fold_results.csv",
        index=False,
    )

    subjects = pd.read_csv(source / "subject_level_summary.csv")
    subjects = subjects[subjects["model_name"] == "exchangeable_block_mixture"].copy()
    if len(subjects) != 24:
        raise AssertionError(
            f"Expected 24 block-mixture subject/version rows, found {len(subjects)}"
        )
    subjects.to_csv(
        source / "exchangeable_block_mixture_subject_summary.csv",
        index=False,
    )


def write_three_model_tables() -> None:
    output = ROOT / "hybrid" / "04_three_model_comparison" / "results"

    random = pd.read_csv(ROOT / "hybrid" / "01_random_fourfold" / "results" / "comparison.csv")
    random.to_csv(output / "random_fourfold_three_model_comparison.csv", index=False)

    chronological = pd.read_csv(
        ROOT
        / "hybrid"
        / "02_chronological_prediction"
        / "results"
        / "overall_comparison.csv"
    )
    chronological.to_csv(
        output / "chronological_four_model_comparison.csv", index=False
    )

    baselines = pd.read_csv(
        ROOT
        / "exchangeable_block_mixture"
        / "results"
        / "comparisons"
        / "overall_model_comparison.csv"
    )
    baselines.to_csv(
        output / "baseline_and_hmm_fourfold_comparison.csv", index=False
    )

    criteria = pd.read_csv(
        ROOT
        / "hybrid"
        / "03_information_criteria"
        / "results"
        / "overall_information_criteria.csv"
    )
    criteria.to_csv(output / "three_model_information_criteria.csv", index=False)

    subject_criteria = pd.read_csv(
        ROOT
        / "hybrid"
        / "03_information_criteria"
        / "results"
        / "subject_information_criteria.csv"
    )
    subject_criteria.to_csv(
        output / "three_model_subject_information_criteria.csv", index=False
    )


def main() -> None:
    write_exchangeable_block_tables()
    write_three_model_tables()
    print("Curated model-specific tables written and row-count checks passed.")


if __name__ == "__main__":
    main()
