from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .circular import deg2rad, circ_diff


@dataclass
class DataBundle:
    df: pd.DataFrame
    raw_n_rows: int
    dropped_n_rows: int
    sequences: list[np.ndarray]
    seq_meta: pd.DataFrame
    y: np.ndarray
    theta: np.ndarray
    prior_mu: np.ndarray
    cos_s: np.ndarray
    cos_p: np.ndarray
    coh_idx: np.ndarray
    prior_idx: np.ndarray
    subject_idx: np.ndarray
    coh_values: np.ndarray
    prior_values: np.ndarray
    subject_values: np.ndarray
    X_transition: np.ndarray
    X_transition_raw: np.ndarray
    transition_means: np.ndarray
    transition_sds: np.ndarray
    transition_names: list[str]

    def fitting_view(self) -> "DataBundle":
        """Return the numerical model inputs without reporting-only frames.

        Process workers do not use ``df`` or ``seq_meta`` during optimization.
        Omitting their blocks keeps deterministic restart parallelism from
        repeatedly serializing the complete trial table on Windows.
        """
        return replace(
            self,
            df=self.df.iloc[0:0].copy(),
            seq_meta=self.seq_meta.iloc[0:0].copy(),
        )


def _first_existing(cols: Iterable[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of the candidate columns were found: {candidates}")


def load_direction_data(csv_path: str | Path) -> DataBundle:
    """Load the Laquitaine & Gardner direction-estimation CSV.

    Expected column names from the public repository include:
    estimate_x, estimate_y, motion_direction, motion_coherence, prior_std,
    prior_mean, subject_id, session_id, run_id, trial_index.
    The loader is deliberately tolerant of minor naming variations.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    raw_n_rows = len(df)
    # Retain the source-file position so downstream audit exports can map every
    # analyzed row back to the unchanged public CSV.
    df["raw_row_index"] = np.arange(raw_n_rows, dtype=int)
    cols = set(df.columns)

    estimate_x_col = _first_existing(cols, ["estimate_x", "resp_x", "response_x"])
    estimate_y_col = _first_existing(cols, ["estimate_y", "resp_y", "response_y"])
    direction_col = _first_existing(cols, ["motion_direction", "direction", "stim_direction", "true_direction"])
    coherence_col = _first_existing(cols, ["motion_coherence", "coherence"])
    prior_std_col = _first_existing(cols, ["prior_std", "prior_sd", "std_prior"])
    prior_mean_col = _first_existing(cols, ["prior_mean", "prior_mu", "mean_prior"])
    subject_col = _first_existing(cols, ["subject_id", "subject", "sub"])
    session_col = _first_existing(cols, ["session_id", "session", "sess"])
    run_col = _first_existing(cols, ["run_id", "run", "block", "block_id"])
    trial_col = _first_existing(cols, ["trial_index", "trial", "trial_id"])

    df = df.dropna(subset=[estimate_x_col, estimate_y_col, direction_col, coherence_col, prior_std_col, prior_mean_col]).copy()
    df = df.rename(columns={
        estimate_x_col: "estimate_x",
        estimate_y_col: "estimate_y",
        direction_col: "motion_direction",
        coherence_col: "motion_coherence",
        prior_std_col: "prior_std",
        prior_mean_col: "prior_mean",
        subject_col: "subject_id",
        session_col: "session_id",
        run_col: "run_id",
        trial_col: "trial_index",
    })

    est_deg = (np.degrees(np.arctan2(df["estimate_y"].to_numpy(), df["estimate_x"].to_numpy())) + 360.0) % 360.0
    df["estimate_deg"] = est_deg
    df = df.sort_values(["subject_id", "session_id", "run_id", "trial_index"]).reset_index(drop=True)

    y = deg2rad(df["estimate_deg"].to_numpy())
    theta = deg2rad(df["motion_direction"].to_numpy())
    prior_mu = deg2rad(df["prior_mean"].to_numpy())
    cos_s = np.cos(circ_diff(y, theta)).astype(float)
    cos_p = np.cos(circ_diff(y, prior_mu)).astype(float)

    coh_values = np.array(sorted(df["motion_coherence"].unique()))
    prior_values = np.array(sorted(df["prior_std"].unique()))
    subject_values = np.array(sorted(df["subject_id"].unique()))
    coh_map = {v: i for i, v in enumerate(coh_values)}
    prior_map = {v: i for i, v in enumerate(prior_values)}
    subject_map = {v: i for i, v in enumerate(subject_values)}
    coh_idx = df["motion_coherence"].map(coh_map).to_numpy(dtype=int)
    prior_idx = df["prior_std"].map(prior_map).to_numpy(dtype=int)
    subject_idx = df["subject_id"].map(subject_map).to_numpy(dtype=int)

    sequences: list[np.ndarray] = []
    metas: list[dict] = []
    group_cols = ["subject_id", "session_id", "run_id"]
    for key, g in df.groupby(group_cols, sort=False):
        idx = g.index.to_numpy(dtype=int)
        sequences.append(idx)
        metas.append({
            "seq_id": len(sequences) - 1,
            "subject_id": key[0],
            "session_id": key[1],
            "run_id": key[2],
            "n": len(idx),
            "prior_std": float(g["prior_std"].iloc[0]),
        })
    seq_meta = pd.DataFrame(metas)

    X_transition_raw, transition_raw_names = build_transition_covariates_raw(df)
    X_transition, transition_means, transition_sds = standardize_transition_covariates(X_transition_raw)
    transition_names = ["intercept"] + transition_raw_names

    return DataBundle(
        df=df,
        raw_n_rows=raw_n_rows,
        dropped_n_rows=raw_n_rows - len(df),
        sequences=sequences,
        seq_meta=seq_meta,
        y=y,
        theta=theta,
        prior_mu=prior_mu,
        cos_s=cos_s,
        cos_p=cos_p,
        coh_idx=coh_idx,
        prior_idx=prior_idx,
        subject_idx=subject_idx,
        coh_values=coh_values,
        prior_values=prior_values,
        subject_values=subject_values,
        X_transition=X_transition,
        X_transition_raw=X_transition_raw,
        transition_means=transition_means,
        transition_sds=transition_sds,
        transition_names=transition_names,
    )


def build_transition_covariates_raw(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build unstandardized row-wise transition covariates.

    The covariate vector for trial t contains information available at or before
    trial t. For a transition from t-1 to t, use X_transition[t]. Sequence starts
    are harmless because transition terms are only used for t>0 within each run.
    """
    theta = deg2rad(df["motion_direction"].to_numpy())
    prior_mu = deg2rad(df["prior_mean"].to_numpy())
    y = deg2rad(df["estimate_deg"].to_numpy())

    coherence = df["motion_coherence"].to_numpy(dtype=float)
    prior_std = df["prior_std"].to_numpy(dtype=float)
    conflict = np.abs(circ_diff(theta, prior_mu))
    current_error = np.abs(circ_diff(y, theta))

    prev_error = np.zeros_like(current_error)
    prev_conflict = np.zeros_like(conflict)
    prev_coherence = np.zeros_like(coherence)
    for _, g in df.groupby(["subject_id", "session_id", "run_id"], sort=False):
        idx = g.index.to_numpy(dtype=int)
        prev_error[idx[0]] = 0.0
        prev_conflict[idx[0]] = 0.0
        prev_coherence[idx[0]] = coherence[idx[0]]
        prev_error[idx[1:]] = current_error[idx[:-1]]
        prev_conflict[idx[1:]] = conflict[idx[:-1]]
        prev_coherence[idx[1:]] = coherence[idx[:-1]]

    prior_precision = 1.0 / np.maximum(prior_std, 1e-6)
    X_raw = np.column_stack([
        coherence,
        prior_precision,
        conflict,
        prev_error,
        prev_conflict,
        prev_coherence,
    ])
    names = ["coherence", "prior_precision", "conflict", "prev_error", "prev_conflict", "prev_coherence"]
    return X_raw.astype(float), names


def standardize_transition_covariates(
    X_raw: np.ndarray,
    means: np.ndarray | None = None,
    sds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize transition covariates and prepend an intercept column."""
    X_raw = np.asarray(X_raw, dtype=float)
    if means is None:
        means = X_raw.mean(axis=0)
    if sds is None:
        sds = X_raw.std(axis=0)
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float).copy()
    sds[sds < 1e-12] = 1.0
    X_z = (X_raw - means) / sds
    X = np.column_stack([np.ones(len(X_raw)), X_z])
    return X.astype(float), means, sds


def transition_scaler_from_sequences(data: DataBundle, seq_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fit transition-covariate standardization on selected sequences only."""
    idx = np.concatenate([data.sequences[int(s)] for s in seq_ids])
    raw = data.X_transition_raw[idx]
    means = raw.mean(axis=0)
    sds = raw.std(axis=0)
    sds[sds < 1e-12] = 1.0
    return means, sds


def with_transition_scaler(data: DataBundle, means: np.ndarray, sds: np.ndarray) -> DataBundle:
    """Return a shallow copy whose transition covariates use the supplied scaler."""
    X, means, sds = standardize_transition_covariates(data.X_transition_raw, means=means, sds=sds)
    return replace(data, X_transition=X, transition_means=means, transition_sds=sds)


def build_transition_covariates(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Backward-compatible all-data standardized transition covariates."""
    X_raw, raw_names = build_transition_covariates_raw(df)
    X, _, _ = standardize_transition_covariates(X_raw)
    return X, ["intercept"] + raw_names


def sequence_labels_for_cv(data: DataBundle) -> np.ndarray:
    return (data.seq_meta["subject_id"].astype(str) + "_p" + data.seq_meta["prior_std"].astype(str)).to_numpy()
