from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


workspace = Path(__file__).resolve().parents[2]
root = workspace / "HMM_fourfold_comparison"
csv_dir = root / "csv"
html_dir = root / "html"
figure_dir = root / "figures"
log_dir = root / "logs"
artifact_dir = root / "model_artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


notebook_path = root / "notebooks" / "HMM_fourfold_heldout_comparison.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]

trials = pd.read_csv(csv_dir / "our_model_fourfold_trial_predictions.csv")
fits = pd.read_csv(csv_dir / "cv_fit_diagnostics.csv")
folds = pd.read_csv(csv_dir / "cv_fold_assignments.csv")
overall = pd.read_csv(csv_dir / "our_model_fourfold_overall_summary.csv").iloc[0]
comparison = pd.read_csv(csv_dir / "heldout_model_comparison.csv")
html_path = html_dir / "HMM_fourfold_comparison_with_original_report.html"
html_text = html_path.read_text(encoding="utf-8")
new_section = html_text.split('<section id="our-fourfold-comparison">', 1)[-1]

before = json.loads((artifact_dir / "protected_snapshot_before.json").read_text(encoding="utf-8"))
after = json.loads((artifact_dir / "protected_snapshot_after.json").read_text(encoding="utf-8"))

relative_assets = re.findall(r'(?:src|href)="(\.\./[^"#]+)"', new_section)
missing_assets = []
for relative in relative_assets:
    target = (html_dir / relative).resolve()
    if not target.exists():
        missing_assets.append(relative)

fold_run_counts = folds.groupby("fold")["cv_group_id"].nunique().sort_index().to_numpy()
our_value = float(overall["mean_held_out_ll_per_trial"])
covariate_value = float(
    comparison.loc[comparison["model"] == "Original HTML: Covariate HMM", "mean_held_out_ll_per_trial"].iloc[0]
)

checks = {
    "notebook_code_cells_executed": all(cell.get("execution_count") is not None for cell in code_cells),
    "notebook_has_zero_error_outputs": len(error_outputs) == 0,
    "trial_count_83210": len(trials) == 83210,
    "all_subjects": trials["subject_id"].nunique() == 12,
    "all_original_runs": trials["cv_group_id"].nunique() == 388,
    "forty_eight_fits": len(fits) == 48,
    "all_fits_converged": fits["converged"].astype(bool).all(),
    "all_fits_monotonic": fits["likelihood_monotonic"].astype(bool).all(),
    "no_duplicate_test_trials": not trials.duplicated("_original_row_index").any(),
    "four_fold_assignments": sorted(folds["fold"].unique().tolist()) == [0, 1, 2, 3],
    "fold_run_counts_balanced": int(fold_run_counts.max() - fold_run_counts.min()) <= 7,
    "prior_predictive_sums_one": np.allclose(
        trials[["prior_predictive_prob_sensory", "prior_predictive_prob_prior", "prior_predictive_prob_lapse"]].sum(axis=1), 1.0, atol=1e-10
    ),
    "filtered_sums_one": np.allclose(
        trials[["filtered_prob_sensory", "filtered_prob_prior", "filtered_prob_lapse"]].sum(axis=1), 1.0, atol=1e-10
    ),
    "overall_ll_reconstructs": np.isclose(
        trials["one_step_predictive_log_likelihood"].sum() / len(trials), our_value, atol=1e-12
    ),
    "comparison_contains_covariate_hmm": np.isclose(covariate_value, -0.8133),
    "combined_html_contains_original_title": "Temporally Persistent Strategy Arbitration" in html_text,
    "combined_html_contains_new_model": "Our revised 3-state subject-specific soft-EM HMM" in html_text,
    "combined_html_contains_prediction_distinction": "Smoothed posterior probabilities are not used for prediction" in html_text,
    "combined_html_relative_assets_exist": len(missing_assets) == 0,
    "png_svg_figure_pairs": all(
        (figure_dir / f"{stem}.png").exists() and (figure_dir / f"{stem}.svg").exists()
        for stem in (
            "heldout_model_comparison",
            "our_model_fold_performance",
            "our_model_subject_performance",
            "cv_convergence_iterations",
            "predictive_density_ratio_vs_source_models",
        )
    ),
    "protected_result_folders_unchanged": before == after,
}
checks = {name: bool(value) for name, value in checks.items()}

lines = [
    "HMM FOUR-FOLD COMPARISON - POST RUN VALIDATION",
    f"notebook_execution_counts={[cell.get('execution_count') for cell in code_cells]}",
    f"notebook_error_outputs={len(error_outputs)}",
    f"trial_count={len(trials)}",
    f"run_count={trials['cv_group_id'].nunique()}",
    f"fold_run_counts={fold_run_counts.tolist()}",
    f"fit_count={len(fits)}",
    f"converged_fits={int(fits['converged'].astype(bool).sum())}",
    f"our_heldout_ll_per_trial={our_value:.12f}",
    f"source_covariate_hmm_ll_per_trial={covariate_value:.4f}",
    f"point_difference_our_minus_covariate={our_value - covariate_value:+.12f}",
]
lines += [f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()]
lines.append(f"overall_status: {'PASS' if all(checks.values()) else 'FAIL'}")
(log_dir / "POST_RUN_VALIDATION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest_rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    if path.name == "output_manifest.csv":
        continue
    if "__pycache__" in path.parts:
        continue
    manifest_rows.append(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )
pd.DataFrame(manifest_rows).to_csv(log_dir / "output_manifest.csv", index=False)

print("\n".join(lines))
if not all(checks.values()):
    raise SystemExit(1)
