# HMM revised executive summary

- **Data:** 12 subjects, 83,210 valid trials; three missing responses removed without imputation or cross-gap transitions.
- **Numerical status:** 12/12 exported; 11/12 met the stopping criterion; Subject 10 remained slowly improving at 600 iterations.
- **State inference:** Smoothed posterior occupancy is descriptive and uses future trials. Filtered probabilities use data through the current trial.
- **Prediction:** Primary performance metric is one-step-ahead pre-response density; mean log predictive density=-0.8021/trial, mean absolute circular error=29.61°.
- **Warnings:** Kappa boundary subjects [5, 6, 11]; transition-floor subjects [1, 2, 3, 4, 5, 6, 7, 8, 9]; no NaN/Inf, state collapse, or lapse collapse.
- **Interpretation:** Patterns are consistent with sensory-, prior-, and lapse-like response modes but do not prove literal psychological strategy switching.

See `reports/` for full details.
