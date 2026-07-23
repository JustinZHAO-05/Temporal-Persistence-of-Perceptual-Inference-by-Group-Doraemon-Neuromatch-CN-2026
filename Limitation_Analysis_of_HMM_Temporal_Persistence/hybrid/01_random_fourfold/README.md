# Random four-fold hybrid evaluation

The executed notebook calls the hybrid implementation in
`../common/markov_test_core.py`, including `fit_hybrid()` and `score_hybrid()`.
Training and test blocks use the same fold assignments as the HMM and
exchangeable block-mixture comparisons.

## Results

| Model | Held-out LL/trial |
|---|---:|
| Exchangeable block mixture | -0.794568 |
| Original three-state HMM | -0.810573 |
| Block + Markov hybrid | -0.787089 |

The hybrid-minus-block difference is +0.007479 LL/trial. Twenty of 48 hybrid
training fits met the strict convergence criterion. The saved
`results/fold_results.csv` and `results/trial_predictions.csv` retain all fits
rather than discarding non-converged results.

The original outputs did not retain fold-level parameter values. The added
`../common/export_hybrid_evaluation_parameters.py random` workflow reproduces
them under `results/parameter_exports/` and checks every saved test likelihood
against the original result.

The comparison indicates better held-out prediction by the fitted hybrid in
this fold design. It does not determine whether the additional predictive
structure corresponds to a unique cognitive mechanism.
