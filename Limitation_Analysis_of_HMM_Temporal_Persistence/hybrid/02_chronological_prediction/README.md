# Chronological forward prediction

This analysis uses expanding windows of earlier blocks to predict immediately
later blocks. It is distinct from random four-fold cross-validation and is
intended to reduce reliance on interleaved train/test periods.

The saved comparison covers 50,896 test trials:

| Model | Chronological LL/trial |
|---|---:|
| Block + Markov hybrid | -0.787565 |
| Exchangeable block mixture | -0.802133 |
| Original HMM | -0.821294 |
| Prior-conditioned IID | -0.837612 |

The hybrid-minus-block difference is +0.014568 LL/trial. Twenty-five of 48
hybrid fits met the strict convergence criterion. A positive held-out
difference supports residual predictive sequence information relative to this
particular block baseline; it does not identify a unique mechanism or guarantee
that every participant exhibits the same effect.

The original outputs did not retain split-level parameter values. The added
`../common/export_hybrid_evaluation_parameters.py chronological` workflow
reproduces them under `results/parameter_exports/` and checks every saved test
likelihood against the original result.
