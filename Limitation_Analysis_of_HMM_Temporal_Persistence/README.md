# Limitation analysis of HMM temporal persistence

## Within-block permutation control and alternative block-level models

This work was developed as part of the 2026 Neuromatch Computational
Neuroscience pod project by Team Doraemon in Pod Jormungandr Dill. We gratefully
thank our TA, Eric Wang, and our Project TA, Arun Kumar, for their guidance and
support. Yongzhang Ji also sincerely thanks his fellow team members Yanzhe Zhao,
Jingnan Ru, Cheryl Yang, Yixuan Li, Xinning Yang, and Deemah Abumelha, each of
whom contributed remarkable care, effort, and commitment throughout the pod
work.

This contribution examines a limitation in the interpretation of temporal
persistence inferred by a three-state hidden Markov model (HMM). Its purpose is
not to reject the HMM as a useful representation of behavior. Instead, it asks
whether a transition matrix with high diagonal probabilities is sufficient
evidence that participants' response strategies follow a genuine first-order,
trial-to-trial Markov process.

The contribution follows two linked tests:

1. A within-block permutation control tests how strongly the HMM evidence
   depends on the observed trial order while preserving block composition.
2. An exchangeable block-IID baseline and a block-plus-Markov hybrid test
   whether longer-timescale block structure can account for predictive patterns
   that might otherwise appear as HMM state persistence.

The directory is a non-destructive, curated copy of the completed analyses. The
original workspace files were not moved, renamed, or deleted. It contains:

1. `HMM/`: the three-state HMM, four-fold held-out evaluation, within-block
   permutation control, and descriptive Viterbi trajectories.
2. `exchangeable_block_mixture/`: a three-class exchangeable block mixture with
   no trial-to-trial latent-state transitions.
3. `hybrid/`: a block-class mixture with a class-specific HMM, evaluated using
   random four-fold and chronological prediction, plus a full-data AIC/BIC
   audit.

The primary predictive metric is held-out one-step-ahead response log predictive
density. Smoothed posteriors and Viterbi paths use information from an entire
sequence and are treated as descriptive, not as held-out predictive evidence.

## 1. Within-block permutation control

Trial order was permuted independently within each valid block segment. The
procedure preserves the trials and their stimulus-response pairings, as well as
the composition of each block, while disrupting the original local ordering.
Under a strong interpretation in which high HMM self-transition probabilities
primarily reflect dependence of the current strategy on the immediately
preceding strategy, this manipulation should markedly alter the fitted
transition structure and impair held-out prediction.

The shuffle reduced the mean fitted diagonal transition probability from
0.6767 to 0.6233. Held-out log predictive density declined from -0.810573 to
-0.831003 per trial, a difference of 0.020430 in favor of the original order;
the direction of this difference was consistent across all 12 participants.
Thus, trial order contained additional predictive information under this
design. However, prediction did not collapse and the shuffled fits remained
diagonal-dominant. High diagonal probabilities therefore indicate that the HMM
represents persistent latent states, but they are not, by themselves, specific
evidence for a genuine first-order psychological switching process.

## 2. Block-level baselines challenge the unique Markov interpretation

The next analysis asks whether a model without trial-to-trial latent-state
transitions can explain the same behavior. The exchangeable block mixture
assigns a latent response class at the block level and treats trials as
exchangeable conditional on that class. Despite lacking Markov transitions, it
outperformed the original HMM in both saved held-out evaluations. This result
shows that longer-timescale block heterogeneity can account for substantial
predictive structure that a trial-level HMM may otherwise express through high
self-transition probabilities.

The hybrid combines a block-level class with class-specific Markov dynamics and
had the highest held-out predictive density among the three fitted models. This
does not establish the hybrid as the true psychological mechanism: several
hybrid fits did not meet the strict convergence criterion, and AIC and BIC did
not select the same model. Taken together, the results support a narrower
interpretation:

- The three HMM states remain useful descriptions of possible response
  strategies in this perceptual-estimation task.
- High HMM self-transition probabilities should not be interpreted alone as
  direct evidence of trial-by-trial psychological state switching.
- The inferred persistence may partly represent stable participant- or
  block-level response tendencies, while the hybrid's predictive improvement
  leaves open the possibility of additional local sequential structure.

## Main numerical results

| Model or control | Evaluation | LL/trial |
|---|---|---:|
| Original three-state HMM | random four-fold | -0.810573 |
| Within-block shuffled HMM | random four-fold | -0.831003 |
| Exchangeable block mixture | random four-fold | -0.794568 |
| Block + Markov hybrid | random four-fold | -0.787089 |
| Exchangeable block mixture | chronological | -0.802133 |
| Block + Markov hybrid | chronological | -0.787565 |

These results challenge a unique first-order Markov interpretation of the
original HMM; they do not show that the HMM states are meaningless or identify
a unique alternative psychological mechanism. The hybrid improved held-out
prediction relative to the fitted exchangeable block baseline, but several
hybrid fits did not satisfy the strict convergence threshold. That limitation
should accompany interpretation of the point estimates.

For the full-data information-criteria audit, AIC favored the hybrid whereas BIC
favored the exchangeable block mixture. The original HMM ranked last under both.
Because only 6 of 12 full-data hybrid fits met the strict convergence threshold,
the information-criteria comparison is secondary to held-out prediction.

## Validation

From this directory, run:

```text
python validate_bundle.py
```

The script checks required files, trial counts, model counts, and the numerical
values quoted above without modifying results.

## Public-release notes

If this contribution is merged, its software is covered by the repository-level
MIT License. The cited source dataset is listed as CC BY 4.0. The complete
transformation history from the deposited MATLAB files to the processed CSV is
not available in this bundle, so that provenance limitation is stated below.
An external HTML benchmark file used during development is intentionally not
included because its redistribution provenance was not documented.

## Data source and citation

The behavioral data used in this analysis were reported by the contributor as
originating from the direction-estimation experiment with four priors released
with:

> Laquitaine, S., & Gardner, J. L. (2018). A switching observer for human
> perceptual estimation. *Neuron, 97*(2), 462-474.e6.
> https://doi.org/10.1016/j.neuron.2017.12.011

The associated public dataset is:

> Laquitaine, S. (2018). A switching observer for human perceptual estimation.
> Laquitaine et al. [Dataset]. Mendeley Data, Version 1.
> https://doi.org/10.17632/nxkvtrj9ps.1

The Mendeley Data record describes the deposited files as experimental blocks
organized by task and participant and lists a CC BY 4.0 license. The CSV files
in this bundle are analysis-ready processed copies; the complete transformation
history from the deposited MATLAB files to these CSV files is not documented
here and should not be inferred from the citation alone.

## Provenance

Copied files retain their original names. Newly written files are English
README documents, path adaptations inside copied code, a bundle validator, and
a hybrid-parameter export script. `FILE_MANIFEST.csv` records the relative path,
byte size, and SHA-256 digest of every other file in the bundle.
