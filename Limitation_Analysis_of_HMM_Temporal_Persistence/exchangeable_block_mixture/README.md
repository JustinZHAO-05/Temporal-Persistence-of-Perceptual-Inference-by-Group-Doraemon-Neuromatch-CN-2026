# Exchangeable block-mixture model

This folder contains the three-class exchangeable block mixture used as the
strongest non-Markov baseline in the saved analyses.

Each sequence segment has a latent block class. Conditional on that class,
trials use fixed sensory, prior, and lapse mixture weights and have no
trial-to-trial latent-state transition. During sequential held-out scoring,
past responses update the posterior over block class causally. The joint block
likelihood is invariant to trial order.

## Code and results

- Public model interface: `code/exchangeable_block_model.py`.
- Original shared source: `code/iid_baseline_core.py`, especially
  `fit_block_mixture()` and the block-mixture branch of `score_model()`.
- Shared HMM emission utilities: `code/fourfold_cv_core.py`.
- Executed notebook: `code/IID_baseline_analysis.ipynb`.
- Full-fit participant results: `results/full_fit/`.
- Trial-level held-out results: `results/heldout/heldout_predictions_*.csv`.
- Aggregate comparisons: `results/comparisons/`.

The generic `full_fit_summary.csv`, `fold_level_results.csv`, and
`subject_level_summary.csv` are migrated source tables containing all three
IID/non-Markov baselines. Clearly named block-mixture-only extracts are:

- `results/full_fit/exchangeable_block_mixture_full_fit_summary.csv`;
- `results/heldout/exchangeable_block_mixture_fold_results.csv`;
- `results/heldout/exchangeable_block_mixture_subject_summary.csv`.

`code/validate_exchangeable_block.py` performs an in-memory participant-1
refit and requires the final likelihood to reproduce the saved full-fit value
within \(10^{-8}\).

The same-fold weighted held-out LL/trial was -0.794568. This was higher than
the saved original HMM estimate (-0.810573) and shuffled HMM estimate
(-0.831003). The block mixture exceeded the original HMM for 11 of 12
participants and the shuffled HMM for 12 of 12 participants. Forty-two of 48
cross-validation fits and 11 of 12 full-data fits met the strict convergence
criterion.

These comparisons show that the fitted exchangeable block model predicted the
held-out responses better under this evaluation. They do not establish that
the data-generating process is exchangeable or that trial-level dependence is
absent.
