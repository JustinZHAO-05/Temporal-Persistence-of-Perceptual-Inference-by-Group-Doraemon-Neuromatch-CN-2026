# Perceptual State Arbitration Results Summary

This report is generated from the analysis tables in this output directory.
The dataset is behavioral, so neural interpretations should be framed as computational hypotheses rather than direct neural evidence.

## Cross-validated model comparison
Best mean held-out log likelihood per trial: **Covariate_HMM** at `-0.8133`.
Static HMM minus independent Switching: `0.0621` LL/trial.
Sequence-bootstrap CI for HMM improvement over independent Switching: `0.0498` to `0.0756` LL/trial.

## Static HMM persistence
Mean fold-level `A_SS`: `0.9371`.
Mean fold-level `A_PP`: `0.9634`.
High values support temporally persistent sensory- and prior-reliance states if held-out model comparisons also favor the HMM.

## Subject-level fits
Subjects fit: `12`.
Mean subject `A_SS`: `0.8674`; range `0.5808` to `0.9781`.
Mean subject `A_PP`: `0.8393`; range `0.0341` to `0.9939`.

## Covariate-HMM effects
- `prior_precision` changes `L_lapse` stay probability by `0.3086` from -1 SD to +1 SD.
- `coherence` changes `P_prior` stay probability by `-0.2144` from -1 SD to +1 SD.
- `coherence` changes `S_sensory` stay probability by `0.2047` from -1 SD to +1 SD.
- `coherence` changes `L_lapse` stay probability by `-0.1811` from -1 SD to +1 SD.
- `conflict` changes `P_prior` stay probability by `-0.1789` from -1 SD to +1 SD.
- `prev_coherence` changes `P_prior` stay probability by `0.1563` from -1 SD to +1 SD.
Previous-error effects should be interpreted conservatively unless they are large and stable.

## Posterior predictive checks
Each final HMM was checked with `100` complete simulated datasets.
**Covariate_HMM:** `19/60` condition-metric cells covered.
- `circular_std_error_deg`: `3/12` cells.
- `mean_abs_error_deg`: `3/12` cells.
- `mean_cos_error`: `4/12` cells.
- `median_abs_error_deg`: `4/12` cells.
- `prior_like_rate`: `5/12` cells.
**HMM_static:** `2/60` condition-metric cells covered.
- `circular_std_error_deg`: `1/12` cells.
- `mean_abs_error_deg`: `0/12` cells.
- `mean_cos_error`: `1/12` cells.
- `median_abs_error_deg`: `0/12` cells.
- `prior_like_rate`: `0/12` cells.
Coverage uses 2.5th-97.5th simulation percentiles. It is an absolute-calibration diagnostic, not a formal hypothesis-test pass/fail decision.
Superior held-out likelihood and improved posterior-predictive coverage answer different questions.

## Decoded sensory/prior-only PPC sensitivity
This supplementary check excludes `L_lapse` trials using the same smoothed forward-backward marginal-MAP rule for observed and simulated responses. It is not Viterbi decoding or an online lapse detector; the complete all-trial PPC above remains primary.
**Covariate_HMM:** coverage improved from `19/60` all-trial cells to `24/60` decoded S/P-only cells.
**HMM_static:** coverage deteriorated from `2/60` all-trial cells to `0/60` decoded S/P-only cells.
**Covariate_HMM retention:** `74246/83210` observed trials; mean `74629.4/83210` simulated trials.
**HMM_static retention:** `79041/83210` observed trials; mean `79015.1/83210` simulated trials.
Removing lapse-classified trials does not automatically establish model adequacy. The all-state run-length PPC is unchanged.

## In-sample information criteria
Lowest AIC: `Covariate_HMM`.
Lowest BIC: `Covariate_HMM`.
Use held-out likelihood as the primary comparison; AIC/BIC are secondary in-sample diagnostics.

## Interpretation rule
A strong behavioral conclusion is supported only if the static or covariate HMM beats independent Switching and serial-dependence baselines in held-out likelihood, and sensory/prior self-transition probabilities remain high across folds and subjects.
