"""Public interface for the exchangeable-block plus class-specific HMM model.

The original functions remain in ``markov_test_core.py`` for provenance. This
module provides a model-specific import surface for the curated repository.
"""
from __future__ import annotations

from markov_test_core import (
    chronological_splits,
    fit_hybrid,
    hybrid_initial,
    load_versions,
    modules,
    score_hybrid,
)


fit = fit_hybrid
score = score_hybrid
initial_parameters = hybrid_initial

__all__ = [
    "fit",
    "score",
    "initial_parameters",
    "load_versions",
    "modules",
    "chronological_splits",
]
