# Three-state HMM analyses

This folder contains the revised subject-specific three-state HMM and three
associated analyses. The latent states use sensory-centered, prior-centered,
and uniform lapse emission components.

- `01_full_data_fit/`: full-data fitting for 12 participants.
- `02_fourfold_cv/`: run-level four-fold held-out evaluation.
- `03_shuffle_control/`: within-block trial-order shuffle control.
- `04_viterbi_trajectory/`: descriptive Viterbi decoding from the full-data fit.

The revised full-data notebook is the reference implementation for the saved
full-fit outputs. The earlier all-subject notebook is retained as provenance but
is not the source of the final 12-participant results.

Held-out one-step-ahead predictive density is the primary prediction metric.
Full-data likelihoods, smoothed posteriors, transition matrices, and Viterbi
paths are descriptive and should not be interpreted as independent evidence of
a causal switching mechanism.
