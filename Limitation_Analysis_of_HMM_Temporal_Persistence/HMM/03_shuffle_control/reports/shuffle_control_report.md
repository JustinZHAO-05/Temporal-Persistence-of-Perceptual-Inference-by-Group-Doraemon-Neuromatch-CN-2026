# Trial-order shuffle control report

## Design and reproducibility

- Model block: `subject_id + session_id + run_id` (389 blocks). `run_id` repeats across subjects, while `experiment_id`, `prior_std`, and `prior_mean` are each fixed within every selected block; adding `experiment_id` would not split any block.
- Missing responses were identified before deletion. Each missing trial created a boundary; shuffling occurred independently inside the resulting 388 segments.
- Master seed: `20260717`. Stable segment number `k` uses seed `20260717 + k`; no Python string hash was used.
- Raw / missing / valid trials: 83,213 / 3 / 83,210. Row-wise permutation preserved every valid trial exactly once.

## Inference definitions

- `p_sensory`, `p_prior`, `p_lapse` are **smoothed state posteriors** from forward-backward and use the whole sequence. They are state inference, not out-of-sample prediction.
- `filtered_prob_*` use responses only through the current trial.
- `prior_predictive_prob_*` and `test_log_predictive_density` are **one-step-ahead predictions** before observing the current response. Every test segment begins from training-derived subject-specific pi.
- The primary predictive result below is based on held-out one-step-ahead log predictive density, not smoothed posteriors or training likelihood.

## Full-data shuffled fit (descriptive)

- Successful / converged subjects: 12/12 and 9/12.
- Full-data training likelihood is descriptive only and is not used to infer generalization.
- Mean diagonal transition changed from 0.6767 to 0.6233; mean switch rate from 0.0583 to 0.1007; mean state dwell time from 23.964 to 20.510 trials.

## Four-fold held-out primary result

- Completed fits: 96/96 (12 subjects × 4 folds × 2 versions); failed folds: 0.
- Training convergence at the unchanged 300-iteration cap: 39/48 original folds and 27/48 shuffled folds (66/96 total). Non-converged folds are retained and flagged rather than silently removed; this is a sensitivity caveat for the numerical estimates.
- Weighted original held-out LL/trial: **-0.810573**.
- Weighted shuffled held-out LL/trial: **-0.831003**.
- Original minus shuffled: **+0.020430** LL/trial.
- Subjects with original > shuffled: **12/12**.

## Quality checks

- No omitted or duplicate held-out predictions; each valid trial is test once per data version.
- Original and shuffled versions use the same saved fold membership, with no original block shared across train and test.
- Test parameters are fixed after training; no test gamma, transition, kappa, or EM update is performed.
- Full-fit and held-out posterior rows sum to one; transition rows sum to one; model outputs are finite. Failures are explicitly stored in `failed_subjects_shuffled.csv` and `failed_folds.csv`.
- NaN / Inf in model-generated outputs: 0 / 0. Mean subject-level transition-matrix Frobenius distance was 0.6404.
- Protected input and existing result hashes are compared before and after execution.

## Conclusion

The original order had better held-out performance, supporting additional predictive information in trial order.

## Outputs

All outputs are under `shuffle_control/`: shuffled data and fold assignments in `data/`, subject fits in `full_fit_results/`, trial-level CV predictions and summaries in `heldout_results/`, comparisons in `comparisons/`, and PNG figures in `figures/`.
