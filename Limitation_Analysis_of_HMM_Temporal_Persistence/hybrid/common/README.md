# Shared hybrid code

- `hybrid_model.py` is the preferred model-specific public interface.
- `markov_test_core.py` is the original multi-test implementation from which
  the hybrid functions were used. Functions unrelated to the curated hybrid
  experiments are retained for provenance and may refer to analyses that are
  not included in this three-model bundle.
- `iid_baseline_core.py` and `fourfold_cv_core.py` supply shared block-mixture,
  circular-emission, and sequence utilities.
- `export_hybrid_evaluation_parameters.py` reproduces and exports all random
  four-fold or chronological hybrid parameters.

The public interface exposes fitting, scoring, initialization, data loading,
and chronological split construction without requiring users to navigate the
multi-test module directly.
