# Four-fold held-out HMM evaluation

The primary implementation is `code/fourfold_cv_core.py`. Folds are assigned at
the run/block level within participant, so training and test runs do not
overlap. Test scoring uses training-derived parameters and one-step-ahead
prediction before observing the current response.

## Saved results

- `results/cv_fold_assignments.csv`: deterministic fold assignments.
- `results/cv_fit_diagnostics.csv`: 48 training fits.
- `results/our_model_fourfold_trial_predictions.csv`: held-out trial scores.
- `results/our_model_fourfold_sequence_scores.csv`: sequence-level scores.
- `results/our_model_fourfold_fold_summary.csv`: four fold summaries.
- `results/our_model_fourfold_subject_summary.csv`: participant summaries.
- `results/our_model_fourfold_overall_summary.csv`: overall estimate.

The saved analysis contains 83,210 held-out trials and reports a mean
one-step-ahead held-out log predictive density of approximately -0.810568 per
trial. All 48 training fits were reported as converged and monotonic.

The external model values in the source HTML are benchmarks rather than
same-fold paired estimates because their original fold identities and
sequence-level scores were unavailable. See
`reports/fourfold_comparison_report.md`.
