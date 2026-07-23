# Exchangeable block mixture plus class-specific HMM

The hybrid model assigns each sequence segment to a latent block class and
places a three-state HMM within each class. It therefore combines slow
block-level heterogeneity with trial-to-trial state transitions.

- `01_random_fourfold/`: same-fold held-out comparison with the HMM and
  exchangeable block mixture.
- `02_chronological_prediction/`: expanding-window prediction of later blocks
  from earlier blocks.
- `03_information_criteria/`: full-data AIC/BIC audit and parameter exports.
- `04_three_model_comparison/`: consolidated comparison across evaluations.
- `common/hybrid_model.py`: the public hybrid-model interface.
- `common/markov_test_core.py`: the original shared Markov-test implementation,
  retained for provenance and used internally by the public interface.

The random four-fold hybrid estimate was -0.787089 LL/trial, +0.007479 relative
to the exchangeable block mixture. The chronological estimate was -0.787565,
+0.014568 relative to the exchangeable block mixture. These are predictive
differences for the fitted models, not proof of a unique latent process.

Convergence was incomplete in several hybrid fits. Twenty of 48 random
four-fold fits, 25 of 48 chronological fits, and 6 of 12 full-data fits met the
strict stopping criterion. This numerical limitation should accompany all
interpretation of the hybrid point estimates and information criteria.
