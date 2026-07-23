# Full-data AIC/BIC audit

The audit compares subject-specific full-data fits of the original HMM,
exchangeable block mixture, and block + Markov hybrid.

| Model | Parameters | AIC rank | Subject-wise BIC rank | Converged subjects |
|---|---:|---:|---:|---:|
| Block + Markov hybrid | 396 | 1 | 2 | 6/12 |
| Exchangeable block mixture | 180 | 2 | 1 | 11/12 |
| Original HMM | 180 | 3 | 3 | 11/12 |

AIC favored the hybrid, whereas both reported BIC conventions favored the
exchangeable block mixture. The original HMM ranked last under both criteria.
Because only 6 of 12 hybrid full-data fits met the strict convergence
criterion, these information-criteria values are approximate and secondary to
held-out comparisons.

The original audit omitted full hybrid parameter values. The added
`code/export_full_fit_parameters.py` reproduces each subject fit and writes
parameters, likelihood histories, and a reproduction check to
`results/parameter_exports/`.
