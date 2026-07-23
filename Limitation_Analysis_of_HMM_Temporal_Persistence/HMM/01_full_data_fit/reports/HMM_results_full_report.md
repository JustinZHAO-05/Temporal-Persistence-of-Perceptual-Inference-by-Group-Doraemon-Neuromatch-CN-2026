# HMM revised full results report

## 1. Data and valid trials

The analysis includes 12 subjects and 83,210 valid trials. Three response-missing trials were excluded without imputation: Subject 4 session 5/run 20/trial 146; Subject 4 session 5/run 23/trial 188; Subject 10 session 3/run 12/trial 1. Missing trials create sequence boundaries; no transition is computed across a missing response.

## 2. Model structure and kappa correction

The model has sensory-, prior-, and lapse-like latent states. Sensory and prior emissions are fixed-mean von Mises densities centered on the current stimulus and block prior mean, respectively; lapse is circular uniform. Subject-specific pi and transition matrices are estimated with soft EM. Fixed-mean kappa is optimized by direct one-dimensional maximization of the gamma-weighted Q function. This replaced a resultant-length approximation that was not a strict M-step for fixed means.

## 3. Three distinct inference/prediction quantities

1. **Smoothed state inference:** `posterior_*` uses the complete segment, including future trials. It supports descriptive latent-state inference, not out-of-sample prediction.
2. **Filtered state inference:** `filtered_prob_*` uses observations through the current response only.
3. **One-step-ahead prediction:** `prior_predictive_prob_*` is computed before the current response from subject pi (segment first trial) or the previous filtered state and transition matrix. `one_step_predictive_likelihood` is the pre-response predictive density and is the primary prediction-quality metric.

## 4. Numerical fitting quality

All 12 subjects were exported. 11/12 met the numerical stopping threshold. Subject 10 did not meet the threshold at 600 iterations but retained a monotonic likelihood trajectory and is marked `did_not_meet_numerical_convergence_at_600_iterations`. No subject contains NaN/Inf, state collapse, or lapse collapse. Kappa boundary warnings: [5, 6, 11]. Transition floor warnings: [1, 2, 3, 4, 5, 6, 7, 8, 9].

## 5. Descriptive state occupancy

Mean subject-level smoothed occupancy was sensory=0.569, prior=0.380, lapse=0.051. These are model-based descriptive allocations using future information within each segment.

## 6. Condition effects

Condition summaries report smoothed posterior occupancy by coherence and prior standard deviation. They are descriptive associations; no group-level inferential tests were conducted.

## 7. Transition dynamics

Transition expected counts were recomputed from xi within segments. Hard-state dwell runs use smoothed-posterior argmax labels and never cross segment boundaries. Implied dwell values derive from 1/(1-A_ii).

## 8. Prediction performance

The mean one-step-ahead log predictive density was -0.8021 per trial. Mean absolute circular one-step prediction error was 29.61 degrees. The mixture direction uses response-before sensory/prior state probabilities; the lapse component contributes zero resultant vector. These are in-sample sequential predictions using fitted subject parameters, not held-out cross-validation.

## 9. Posterior confidence and classification-style checks

Posterior confidence, closest-reference matching, and high-confidence examples use smoothed posterior probabilities. They evaluate internal state interpretation and should not be described as prospective prediction accuracy.

## 10. Comparison with the source HTML models

The source report contains held-out comparisons among independent, serial, static-HMM, and covariate-HMM models. Our revised subject-specific soft-EM HMM is added as a documented in-sample model artifact. Its one-step-ahead values are not directly comparable with held-out scores in the source page, so no new numerical ranking is claimed.

## 11. What the results can support

The fitted latent states show emission and condition patterns consistent with sensory-, prior-, and lapse-like response modes. The results support descriptive statements about fitted state occupancy, persistence, condition association, and in-sample sequential predictive density.

## 12. What the results cannot support

The model does not prove that participants literally switch among three psychological strategies, establish causal mechanisms, provide held-out generalization evidence, or justify group-level psychological inference. Subject 10 remains numerically unfinished, and selected boundary parameters warrant caution.

## 13. Next steps

Prioritize held-out or cross-validated model comparison, sensitivity analyses for boundary parameters, and a targeted Subject 10 convergence study before substantive psychological interpretation.
