"""Refit participant 1 in memory to validate the curated block-mixture code."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import exchangeable_block_model as model


def main() -> None:
    original, _ = model.load_data(write_audit=False)
    subject = original[original["subject_id"] == 1].reset_index(drop=True)
    fit = model.fit(subject)
    observed = float(fit["history"][-1])

    root = Path(__file__).resolve().parents[1]
    reference = pd.read_csv(root / "results" / "heldout" / "full_fit_summary.csv")
    expected = float(
        reference[
            (reference["model_name"] == "exchangeable_block_mixture")
            & (reference["subject_id"] == 1)
        ]["final_log_likelihood"].iloc[0]
    )
    difference = observed - expected
    if not np.isclose(observed, expected, atol=1e-8, rtol=0):
        raise AssertionError(
            f"Participant 1 likelihood mismatch: observed={observed}, "
            f"expected={expected}, difference={difference}"
        )
    print(
        {
            "subject_id": 1,
            "n_trials": len(subject),
            "final_log_likelihood": observed,
            "reference_final_log_likelihood": expected,
            "difference": difference,
            "matches_within_1e-8": True,
        }
    )


if __name__ == "__main__":
    main()
