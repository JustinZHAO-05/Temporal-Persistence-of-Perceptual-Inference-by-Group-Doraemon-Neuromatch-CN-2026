# Revised full-data three-state HMM

## Scope

`code/HMM_main_me_diagnostics_and_rerun.ipynb` is the source of the final saved
full-data results. It fits each participant separately, treats missing responses
as sequence boundaries, and uses sensory-centered, prior-centered, and uniform
lapse emissions.

Two earlier files are retained for provenance:

- `HMM_main_me_all_subjects_run.ipynb` was the first all-subject run and did not
  successfully export two participants.
- `HMM_main_me_param_only_checked.ipynb` contains an earlier checked function
  implementation but not the final all-subject workflow.

## Data and outputs

- Input: `data/data01_direction4priors.csv`.
- Raw trials: 83,213.
- Missing-response trials excluded: 3.
- Valid trials: 83,210 across 12 participants.
- Participant outputs: `results/subject_XX/`.
- Group summaries: `results/all_subject_summary_revised.csv` and
  `results/all_subject_transition_matrices_revised.csv`.

All 12 participants were exported. Eleven met the numerical stopping criterion.
Participant 10 remained monotonically improving at the 600-iteration cap and is
flagged accordingly. Boundary and transition-floor warnings are documented in
`reports/HMM_results_quality_control.md`.

## Interpretation boundary

The full-data likelihood and smoothed state posteriors use the observed
sequence. They describe the fitted model but do not measure out-of-sample
prediction. State labels reflect emission components and do not establish
literal psychological strategies.
