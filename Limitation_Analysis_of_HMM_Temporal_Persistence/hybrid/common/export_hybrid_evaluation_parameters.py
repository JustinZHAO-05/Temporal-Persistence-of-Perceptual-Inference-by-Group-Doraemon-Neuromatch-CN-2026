"""Export hybrid parameters for the saved random-fold and chronological fits."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


COMMON = Path(__file__).resolve().parent
ROOT = COMMON.parents[1]
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from hybrid_model import (  # noqa: E402
    chronological_splits,
    fit as fit_hybrid,
    load_versions,
    modules,
    score as score_hybrid,
)


def jsonable(parameters: dict) -> dict:
    return {
        "rho": np.asarray(parameters["rho"], dtype=float).tolist(),
        "pi": np.asarray(parameters["pi"], dtype=float).tolist(),
        "A": np.asarray(parameters["A"], dtype=float).tolist(),
        "kappaS": {str(key): float(value) for key, value in parameters["kappaS"].items()},
        "kappaP": {str(key): float(value) for key, value in parameters["kappaP"].items()},
    }


def fit_and_export(task: dict, base, output: Path) -> dict:
    subject_id = int(task["subject_id"])
    unit = int(task["unit"])
    label = task["label"]
    subject_dir = output / f"subject_{subject_id:02d}" / f"{label}_{unit}"
    parameters_path = subject_dir / "parameters.json"
    history_path = subject_dir / "log_likelihood_history.csv"
    summary_path = subject_dir / "fit_summary.json"

    if parameters_path.exists() and history_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train = task["train"].reset_index(drop=True)
    test = task["test"].copy()
    fit = fit_hybrid(train, base)
    scored = score_hybrid(test, fit["params"], base)
    history = np.asarray(fit["history"], dtype=float)
    result = {
        "subject_id": subject_id,
        label: unit,
        "n_train_trials": int(len(train)),
        "n_test_trials": int(len(test)),
        "converged": bool(fit["converged"]),
        "n_iterations": int(fit["n_iterations"]),
        "train_ll": float(history[-1]),
        "test_total_ll": float(scored["test_log_predictive_density"].sum()),
        "test_ll_per_trial": float(scored["test_log_predictive_density"].mean()),
        "monotonic": bool(fit["monotonic"]),
    }
    subject_dir.mkdir(parents=True, exist_ok=True)
    parameters_path.write_text(json.dumps(jsonable(fit["params"]), indent=2), encoding="utf-8")
    pd.DataFrame(
        {"iteration": np.arange(1, len(history) + 1), "log_likelihood": history}
    ).to_csv(history_path, index=False)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def random_tasks(original: pd.DataFrame) -> list[dict]:
    tasks = []
    for subject_id in sorted(original["subject_id"].unique()):
        subject = original[original["subject_id"] == subject_id]
        for fold_id in range(1, 5):
            tasks.append(
                {
                    "subject_id": int(subject_id),
                    "unit": fold_id,
                    "label": "fold_id",
                    "train": subject[subject["fold_id"] != fold_id].copy(),
                    "test": subject[subject["fold_id"] == fold_id].copy(),
                }
            )
    return tasks


def chronological_tasks(original: pd.DataFrame) -> list[dict]:
    assignments = chronological_splits(original)
    tasks = []
    for subject_id in sorted(original["subject_id"].unique()):
        subject = original[original["subject_id"] == subject_id]
        for split_id in range(1, 5):
            rows = assignments[
                (assignments["subject_id"] == subject_id)
                & (assignments["split_id"] == split_id)
            ]
            train_ids = set(rows[rows["role"] == "train"]["original_block_id"])
            test_ids = set(rows[rows["role"] == "test"]["original_block_id"])
            tasks.append(
                {
                    "subject_id": int(subject_id),
                    "unit": split_id,
                    "label": "split_id",
                    "train": subject[subject["original_block_id"].isin(train_ids)].copy(),
                    "test": subject[subject["original_block_id"].isin(test_ids)].copy(),
                }
            )
    return tasks


def run(mode: str, workers: int) -> pd.DataFrame:
    original, _ = load_versions()
    base, _ = modules()
    if mode == "random":
        experiment = ROOT / "hybrid" / "01_random_fourfold"
        tasks = random_tasks(original)
        reference = pd.read_csv(experiment / "results" / "fold_results.csv")
        keys = ["subject_id", "fold_id"]
        output = experiment / "results" / "parameter_exports"
    else:
        experiment = ROOT / "hybrid" / "02_chronological_prediction"
        tasks = chronological_tasks(original)
        reference = pd.read_csv(experiment / "results" / "chronological_results.csv")
        reference = reference[reference["model_name"] == "block_plus_markov"].copy()
        keys = ["subject_id", "split_id"]
        output = experiment / "results" / "parameter_exports"

    output.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        rows = [fit_and_export(task, base, output) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(lambda task: fit_and_export(task, base, output), tasks))

    summary = pd.DataFrame(rows).sort_values(keys)
    checked = summary.merge(
        reference[keys + ["n_iterations", "test_total_ll"]].rename(
            columns={
                "n_iterations": "reference_n_iterations",
                "test_total_ll": "reference_test_total_ll",
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    checked["test_ll_difference"] = checked["test_total_ll"] - checked["reference_test_total_ll"]
    checked["matches_reference_within_1e-8"] = checked["test_ll_difference"].abs() <= 1e-8
    checked.to_csv(output / "parameter_export_summary.csv", index=False)
    if len(checked) != 48:
        raise AssertionError(f"Expected 48 {mode} fits, found {len(checked)}")
    if not checked["matches_reference_within_1e-8"].all():
        raise AssertionError(f"{mode} test likelihoods do not reproduce saved results")
    print(
        {
            "mode": mode,
            "fits": len(checked),
            "converged": int(checked["converged"].sum()),
            "maximum_absolute_test_ll_difference": float(
                checked["test_ll_difference"].abs().max()
            ),
        }
    )
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["random", "chronological"])
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    run(arguments.mode, max(1, arguments.workers))
