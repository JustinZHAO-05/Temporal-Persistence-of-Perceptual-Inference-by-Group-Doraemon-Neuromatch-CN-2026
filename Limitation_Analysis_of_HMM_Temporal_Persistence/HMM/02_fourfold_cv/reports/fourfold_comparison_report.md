# Four-fold held-out comparison

## Our model

**Our revised 3-state subject-specific soft-EM HMM** was refitted independently in four run-level folds. Training and test runs never overlap. Prediction on held-out segments uses training-derived π at the first trial and filtering thereafter.

## Primary result

- Held-out trials: 83,210
- Original runs: 388
- Mean held-out one-step-ahead log predictive density: -0.810568 per trial
- 95% run-bootstrap interval: [-0.860651, -0.761470]
- Difference from source Covariate HMM point estimate (-0.8133): +0.002732 LL/trial
- Difference from derived source Static HMM point estimate (-0.8431): +0.032532 LL/trial

## Important comparison limitation

The source HTML reports four-fold results but does not contain the original fold IDs or per-sequence scores. This analysis creates and saves new deterministic folds. Therefore the plotted source-model values are external benchmarks, not same-fold paired estimates. A definitive paired bootstrap requires either the original fold assignments and sequence scores or refitting all source models on the new folds.

## Numerical quality

- Converged training-fold fits: 48/48
- Monotonic training-fold fits: 48/48
- Runtime: 276.1 seconds
