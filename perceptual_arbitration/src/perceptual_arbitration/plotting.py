from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hmm import STATE_NAMES


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cv_comparison(cv_summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if cv_summary.empty:
        return
    df = cv_summary.sort_values("mean_test_ll_per_trial", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(df["model"], df["mean_test_ll_per_trial"], xerr=df.get("se_test_ll_per_trial"), color="#4c78a8")
    ax.set_xlabel("Held-out log likelihood per trial")
    ax.set_ylabel("")
    ax.set_title("Cross-validated model comparison")
    ax.grid(axis="x", alpha=0.25)
    _finish(fig, path)


def plot_transition_heatmap(A: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(A, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(STATE_NAMES)), STATE_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(STATE_NAMES)), STATE_NAMES)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Previous state")
    ax.set_title("Static HMM transition matrix")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", color="white" if A[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finish(fig, path)


def plot_subject_transitions(subject_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if subject_df.empty or not {"A_SS", "A_PP"}.issubset(subject_df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(subject_df))
    ax.scatter(x - 0.08, subject_df["A_SS"], label="A_SS", color="#4c78a8")
    ax.scatter(x + 0.08, subject_df["A_PP"], label="A_PP", color="#f58518")
    ax.axhline(1 / 3, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x, subject_df["subject_id"].astype(str), rotation=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Self-transition probability")
    ax.set_title("Subject-level persistence")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def plot_covariate_effects(effects_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if effects_df.empty:
        return
    stay = effects_df[effects_df["previous_state"] == effects_df["next_state"]].copy()
    if stay.empty:
        return
    stay["label"] = stay["covariate"] + " -> " + stay["previous_state"].str.replace("_", " ", regex=False)
    stay = stay.sort_values("delta_plus_minus")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(stay))))
    colors = np.where(stay["delta_plus_minus"] >= 0, "#54a24b", "#e45756")
    ax.barh(stay["label"], stay["delta_plus_minus"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Change in stay probability (+1 SD minus -1 SD)")
    ax.set_ylabel("")
    ax.set_title("Covariate-HMM persistence effects")
    ax.grid(axis="x", alpha=0.25)
    _finish(fig, path)


def plot_posterior_predictive_summary(ppc_summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if ppc_summary.empty:
        return
    df = ppc_summary.groupby(["source", "motion_coherence", "prior_std"], as_index=False)["mean_abs_error_deg"].mean()
    obs = df[df["source"] == "observed"]
    sim = df[df["source"] == "simulated"]
    if obs.empty or sim.empty:
        return
    obs = obs.sort_values(["motion_coherence", "prior_std"])
    sim = sim.sort_values(["motion_coherence", "prior_std"])
    labels = [f"{c:g}/{int(p)}" for c, p in zip(obs["motion_coherence"], obs["prior_std"])]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, obs["mean_abs_error_deg"], marker="o", label="Observed", color="#4c78a8")
    ax.plot(x, sim["mean_abs_error_deg"], marker="o", label="Simulated", color="#f58518")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Mean absolute error (deg)")
    ax.set_xlabel("Coherence / prior SD")
    ax.set_title("Posterior predictive condition summary")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def plot_run_lengths(run_summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if run_summary.empty:
        return
    df = run_summary.pivot_table(index="state", columns="source", values="mean_run_length", aggfunc="mean").reindex(STATE_NAMES)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    df.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Mean run length")
    ax.set_xlabel("")
    ax.set_title("Observed and simulated state persistence")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def plot_serial_alphas(cv_results: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if cv_results.empty or "model" not in cv_results.columns:
        return
    df = cv_results[cv_results["model"].astype(str).str.startswith("Serial_")].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    ax.scatter(x, df.get("alpha_stim", pd.Series(np.nan, index=df.index)), label="alpha_stim", color="#4c78a8")
    ax.scatter(x, df.get("alpha_resp", pd.Series(np.nan, index=df.index)), label="alpha_resp", color="#f58518")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, df["model"].str.replace("Serial_", "", regex=False).str.replace("_independent_switching", "", regex=False) + " f" + df["fold"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Serial attraction coefficient")
    ax.set_title("Serial-dependence baseline fits")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def generate_figures(
    out_dir: str | Path,
    cv_summary: pd.DataFrame | None = None,
    cv_results: pd.DataFrame | None = None,
    transition_matrix: np.ndarray | None = None,
    subject_df: pd.DataFrame | None = None,
    covariate_effects: pd.DataFrame | None = None,
    ppc_summary: pd.DataFrame | None = None,
    run_summary: pd.DataFrame | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    made: list[Path] = []
    jobs = [
        (cv_summary is not None, lambda: plot_cv_comparison(cv_summary, fig_dir / "cv_model_comparison.png"), fig_dir / "cv_model_comparison.png"),
        (transition_matrix is not None, lambda: plot_transition_heatmap(transition_matrix, fig_dir / "hmm_transition_matrix.png"), fig_dir / "hmm_transition_matrix.png"),
        (subject_df is not None, lambda: plot_subject_transitions(subject_df, fig_dir / "subject_transition_persistence.png"), fig_dir / "subject_transition_persistence.png"),
        (covariate_effects is not None, lambda: plot_covariate_effects(covariate_effects, fig_dir / "covariate_hmm_effects.png"), fig_dir / "covariate_hmm_effects.png"),
        (ppc_summary is not None, lambda: plot_posterior_predictive_summary(ppc_summary, fig_dir / "posterior_predictive_condition_summary.png"), fig_dir / "posterior_predictive_condition_summary.png"),
        (run_summary is not None, lambda: plot_run_lengths(run_summary, fig_dir / "state_run_lengths.png"), fig_dir / "state_run_lengths.png"),
        (cv_results is not None, lambda: plot_serial_alphas(cv_results, fig_dir / "serial_dependence_alphas.png"), fig_dir / "serial_dependence_alphas.png"),
    ]
    for enabled, fn, path in jobs:
        if enabled:
            fn()
            if path.exists() and path.stat().st_size > 0:
                made.append(path)
    return made
