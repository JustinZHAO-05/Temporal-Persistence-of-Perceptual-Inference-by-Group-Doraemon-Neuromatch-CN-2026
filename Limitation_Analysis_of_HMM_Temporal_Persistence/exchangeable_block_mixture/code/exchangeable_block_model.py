"""Public interface for the exchangeable block-mixture implementation.

The original analysis implemented three IID/non-Markov baselines in one module.
This file exposes only the exchangeable block-mixture functions so that the
model name and import surface are unambiguous.
"""
from __future__ import annotations

import pandas as pd

from iid_baseline_core import (
    base_module,
    fit_block_mixture,
    load_data,
    score_model,
)


MODEL_NAME = "exchangeable_block_mixture"


def fit(data: pd.DataFrame, base=None) -> dict:
    """Fit the three-class exchangeable block mixture."""
    return fit_block_mixture(
        data.reset_index(drop=True),
        base if base is not None else base_module(),
    )


def score(data: pd.DataFrame, parameters: dict, base=None) -> pd.DataFrame:
    """Return causal sequential scores with no trial-to-trial state transition."""
    return score_model(
        MODEL_NAME,
        data,
        parameters,
        base if base is not None else base_module(),
    )


__all__ = ["MODEL_NAME", "fit", "score", "load_data"]
