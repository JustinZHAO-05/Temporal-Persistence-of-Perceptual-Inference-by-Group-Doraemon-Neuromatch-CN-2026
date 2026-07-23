"""Refit the full-data hybrid model and export parameters omitted by the original audit.

This script uses the same deterministic model implementation, 83,210 valid trials,
300-iteration cap, tolerance, and smoothing constant as the original analysis.
It writes only inside ``hybrid/03_information_criteria/results/parameter_exports``.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
CONTRIBUTION_ROOT = HERE.parents[3]
COMMON = CONTRIBUTION_ROOT / "hybrid" / "common"
OUTPUT = HERE.parents[1] / "results" / "parameter_exports"
REFERENCE_SUMMARY = HERE.parents[1] / "results" / "hybrid_full_fit_summary.csv"

if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from hybrid_model import fit as fit_hybrid, load_versions, modules  # noqa: E402


def _jsonable_parameters(parameters: dict) -> dict:
    return {
        "rho": np.asarray(parameters["rho"], dtype=float).tolist(),
        "pi": np.asarray(parameters["pi"], dtype=float).tolist(),
        "A": np.asarray(parameters["A"], dtype=float).tolist(),
        "kappaS": {str(key): float(value) for key, value in parameters["kappaS"].items()},
        "kappaP": {str(key): float(value) for key, value in parameters["kappaP"].items()},
    }


def _fit_subject(item: tuple[int, pd.DataFrame], base) -> dict:
    subject_id, data = item
    subject_dir = OUTPUT / f"subject_{subject_id:02d}"
    parameter_path = subject_dir / "parameters.json"
    history_path = subject_dir / "log_likelihood_history.csv"
    result_path = subject_dir / "fit_summary.json"

    if parameter_path.exists() and history_path.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    fit = fit_hybrid(data.reset_index(drop=True), base)
    history = np.asarray(fit["history"], dtype=float)
    parameters = _jsonable_parameters(fit["params"])
    result = {
        "subject_id": int(subject_id),
        "n_trials": int(len(data)),
        "converged": bool(fit["converged"]),
        "n_iterations": int(fit["n_iterations"]),
        "final_log_likelihood": float(history[-1]),
        "monotonic": bool(fit["monotonic"]),
    }

    subject_dir.mkdir(parents=True, exist_ok=True)
    parameter_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    pd.DataFrame(
        {"iteration": np.arange(1, len(history) + 1), "log_likelihood": history}
    ).to_csv(history_path, index=False)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(workers: int = 1) -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original, _ = load_versions()
    base, _ = modules()
    items = [
        (int(subject_id), data.copy())
        for subject_id, data in original.groupby("subject_id", sort=True)
    ]

    if workers == 1:
        rows = [_fit_subject(item, base) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(lambda item: _fit_subject(item, base), items))

    summary = pd.DataFrame(rows).sort_values("subject_id")
    reference = pd.read_csv(REFERENCE_SUMMARY).rename(
        columns={
            "final_log_likelihood": "reference_final_log_likelihood",
            "converged": "reference_converged",
            "n_iterations": "reference_n_iterations",
        }
    )
    checked = summary.merge(
        reference[
            [
                "subject_id",
                "reference_final_log_likelihood",
                "reference_converged",
                "reference_n_iterations",
            ]
        ],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    checked["log_likelihood_difference"] = (
        checked["final_log_likelihood"] - checked["reference_final_log_likelihood"]
    )
    checked["matches_reference_within_1e-8"] = (
        checked["log_likelihood_difference"].abs() <= 1e-8
    )
    checked.to_csv(OUTPUT / "parameter_export_summary.csv", index=False)

    if len(checked) != 12:
        raise AssertionError(f"Expected 12 subjects, found {len(checked)}")
    if not np.isfinite(checked["final_log_likelihood"]).all():
        raise AssertionError("Non-finite full-fit likelihood")
    if not checked["matches_reference_within_1e-8"].all():
        raise AssertionError("Refitted likelihoods do not reproduce the original audit")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = main(workers=max(1, args.workers))
    print(result.to_string(index=False))
