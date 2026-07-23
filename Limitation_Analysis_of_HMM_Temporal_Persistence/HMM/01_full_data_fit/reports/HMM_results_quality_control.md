# HMM revised quality-control report

## Numerical success

- Exported subjects: 12/12
- Valid trials: 83,210
- Converged subjects: 11/12
- Nonconverged: [10]
- NaN/Inf: none
- Posterior rows normalized: yes
- Filtered rows normalized: yes
- Prior-predictive rows normalized: yes
- Transition and pi rows normalized: yes
- Observed likelihood monotonic: yes for all subjects

## Boundary warnings

- Kappa near optimizer lower bound: [5, 6, 11]
- Transition probability at/below 1e-6 warning threshold: [1, 2, 3, 4, 5, 6, 7, 8, 9]

## Prediction terminology

- `posterior_*`: smoothed, full-segment state inference.
- `filtered_prob_*`: current-and-past state inference.
- `prior_predictive_prob_*`: response-before state prediction.
- `one_step_predictive_likelihood`: primary sequential prediction metric.

## Remaining limitation

Subject 10 is retained and labeled `did_not_meet_numerical_convergence_at_600_iterations`. All its trial-level outputs are present, but its parameters should be treated as provisional.
