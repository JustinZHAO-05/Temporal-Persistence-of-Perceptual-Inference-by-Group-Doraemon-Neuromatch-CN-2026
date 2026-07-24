from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import DataBundle
from .diagnostics import (
    behavioral_condition_summary,
    dataset_summary,
    observed_response_histograms,
    posterior_state_occupancy,
    ppc_histogram_intervals,
    ppc_metric_intervals,
    representative_sequence,
    run_length_calibration,
)
from .exposition import (
    build_teaching_tables,
    embedded_svg_figure,
    key_result_math_panel,
    model_math_panels,
    teaching_claims,
    validate_exposition_coverage,
    visual_explanation,
)
from .hmm import STATE_NAMES
from .reporting import generate_results_summary
from .run_metadata import atomic_write_json, utc_now


BLUE = "#2F6690"
ORANGE = "#D97706"
OLIVE = "#71893F"
PINK = "#B64E74"
INK = "#252A31"
MUTED = "#6B7280"
GRID = "#D9DEE5"
LIGHT_BLUE = "#C9D8E6"

MODEL_LABELS = {
    "Covariate_HMM": "Covariate HMM",
    "HMM_static": "Static HMM",
    "Independent_switching": "Independent Switching",
    "Serial_stim_independent_switching": "Serial stimulus",
    "Serial_resp_independent_switching": "Serial response",
    "Serial_both_independent_switching": "Serial stimulus + response",
}


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, report_data_dir: Path, name: str) -> Path:
    path = report_data_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _transition_long(out_dir: Path) -> pd.DataFrame:
    matrix = pd.read_csv(out_dir / "hmm_final_transition_matrix.csv", index_col=0)
    rows = []
    for previous in matrix.index:
        for next_state in matrix.columns:
            rows.append({
                "previous_state": previous,
                "next_state": next_state,
                "probability": float(matrix.loc[previous, next_state]),
            })
    return pd.DataFrame(rows)


def _emission_table(out_dir: Path) -> pd.DataFrame:
    params = _read_csv(out_dir / "hmm_final_parameters.csv")
    result = params[params["parameter"].isin(["sensory_kappa", "prior_kappa"])].copy()
    result["family"] = result["parameter"].map({"sensory_kappa": "Sensory", "prior_kappa": "Prior"})
    result["condition_label"] = result["condition"].str.replace("coherence_", "Coherence ", regex=False).str.replace("prior_std_", "Prior SD ", regex=False)
    return result[["parameter", "family", "condition", "condition_label", "value"]]


def _subject_long(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in subject.iterrows():
        for state, value_col, baseline_col, excess_col in [
            ("Sensory", "A_SS", "stationary_S", "persistence_excess_S"),
            ("Prior", "A_PP", "stationary_P", "persistence_excess_P"),
        ]:
            baseline = row.get(baseline_col, np.nan)
            value = row[value_col]
            rows.append({
                "subject_id": int(row["subject_id"]),
                "state": state,
                "self_transition": float(value),
                "stationary_probability": float(baseline) if pd.notna(baseline) else np.nan,
                "persistence_excess": float(row.get(excess_col, value - baseline)) if pd.notna(baseline) else np.nan,
                "converged": bool(row["converged"]),
                "n_trials": int(row["n_trials"]),
            })
    return pd.DataFrame(rows)


def _restart_summary(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ["restart_diagnostics_cv.csv", "restart_diagnostics_final.csv", "restart_diagnostics_subject.csv"]:
        frame = _read_csv(out_dir / name, required=False)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    restarts = pd.concat(frames, ignore_index=True)
    group_cols = ["stage", "model", "fold", "subject_id"]
    summary = restarts.groupby(group_cols, dropna=False).agg(
        n_restarts=("seed", "size"),
        n_converged=("converged", "sum"),
        best_train_ll=("train_ll", "max"),
        worst_train_ll=("train_ll", "min"),
    ).reset_index()
    selected = restarts[restarts["selected"].astype(bool)][group_cols + ["seed", "converged", "n_iter"]].rename(columns={
        "seed": "selected_seed",
        "converged": "selected_converged",
        "n_iter": "selected_n_iter",
    })
    summary = summary.merge(selected, on=group_cols, how="left")
    summary["ll_range"] = summary["best_train_ll"] - summary["worst_train_ll"]
    return restarts, summary


def prepare_report_tables(data: DataBundle, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    report_data_dir = out_dir / "report" / "data"
    report_data_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    tables["dataset_summary"] = dataset_summary(data)
    tables["behavioral_conditions"] = behavioral_condition_summary(data)
    tables["observed_histograms"] = observed_response_histograms(data)
    tables["cv_summary"] = _read_csv(out_dir / "cv_summary.csv")
    tables["cv_summary"]["model_label"] = tables["cv_summary"]["model"].map(MODEL_LABELS).fillna(tables["cv_summary"]["model"])
    tables["bootstrap"] = _read_csv(out_dir / "bootstrap_model_differences.csv")
    tables["bootstrap"]["model_label"] = tables["bootstrap"]["model"].map(MODEL_LABELS).fillna(tables["bootstrap"]["model"])
    tables["transition"] = _transition_long(out_dir)
    tables["emissions"] = _emission_table(out_dir)
    tables["subject"] = _subject_long(_read_csv(out_dir / "subject_level_hmm.csv"))

    posterior = _read_csv(out_dir / "posterior_states.csv")
    tables["occupancy"] = posterior_state_occupancy(posterior)
    sequence, sequence_meta = representative_sequence(posterior, _read_csv(out_dir / "per_sequence_cv_results.csv"))
    tables["representative_sequence"] = sequence
    tables["representative_sequence_metadata"] = sequence_meta

    cv_results = _read_csv(out_dir / "cv_results.csv")
    serial = cv_results[cv_results["model"].str.startswith("Serial_", na=False)].copy()
    serial["model_label"] = serial["model"].map(MODEL_LABELS).fillna(serial["model"])
    tables["serial"] = serial

    covariate = _read_csv(out_dir / "covariate_hmm_effect_intervals.csv", required=False)
    if covariate.empty:
        covariate = _read_csv(out_dir / "covariate_hmm_effects.csv", required=False)
    if not covariate.empty:
        covariate["is_stay"] = covariate["previous_state"] == covariate["next_state"]
        covariate["effect_label"] = covariate["covariate"].str.replace("_", " ") + " -> " + covariate["previous_state"].str.replace("_", " ")
    tables["covariate"] = covariate
    tables["covariate_fold"] = _read_csv(out_dir / "covariate_hmm_fold_effects.csv", required=False)

    combined_metrics = _read_csv(out_dir / "posterior_predictive_model_metric_intervals.csv", required=False)
    combined_histograms = _read_csv(out_dir / "posterior_predictive_model_histogram_intervals.csv", required=False)
    combined_runs = _read_csv(out_dir / "posterior_predictive_model_run_length_calibration.csv", required=False)
    combined_coverage = _read_csv(out_dir / "posterior_predictive_model_coverage.csv", required=False)
    if combined_metrics.empty:
        ppc_summary = _read_csv(out_dir / "posterior_predictive_condition_summary.csv")
        combined_metrics = ppc_metric_intervals(ppc_summary)
        combined_metrics.insert(0, "model", "HMM_static")
    if combined_histograms.empty:
        ppc_hist = _read_csv(out_dir / "posterior_predictive_histograms.csv")
        combined_histograms = ppc_histogram_intervals(ppc_hist)
        combined_histograms.insert(0, "model", "HMM_static")
    if combined_runs.empty:
        run_lengths = _read_csv(out_dir / "state_run_lengths.csv")
        combined_runs = run_length_calibration(run_lengths)
        combined_runs.insert(0, "model", "HMM_static")
    if combined_coverage.empty:
        coverage_source = combined_metrics.copy()
        coverage_source["covered"] = (
            (coverage_source["observed"] >= coverage_source["simulated_ci_low"])
            & (coverage_source["observed"] <= coverage_source["simulated_ci_high"])
        )
        combined_coverage = coverage_source.groupby(["model", "metric"], as_index=False).agg(
            cells=("covered", "size"), covered=("covered", "sum")
        )
        combined_coverage["coverage_rate"] = combined_coverage["covered"] / combined_coverage["cells"]
    tables["ppc_metrics"] = combined_metrics
    tables["ppc_histograms"] = combined_histograms
    tables["run_length_calibration"] = combined_runs
    tables["ppc_coverage"] = combined_coverage

    sp_metrics = _read_csv(out_dir / "posterior_predictive_sp_model_metric_intervals.csv", required=False)
    sp_histograms = _read_csv(out_dir / "posterior_predictive_sp_model_histogram_intervals.csv", required=False)
    sp_coverage = _read_csv(out_dir / "posterior_predictive_sp_model_coverage.csv", required=False)
    sp_retention = _read_csv(out_dir / "posterior_predictive_sp_model_retention_summary.csv", required=False)
    sp_classification = _read_csv(
        out_dir / "posterior_predictive_sp_model_classification_confusion.csv",
        required=False,
    )
    tables["sp_ppc_metrics"] = sp_metrics
    tables["sp_ppc_histograms"] = sp_histograms
    tables["sp_ppc_coverage"] = sp_coverage
    tables["sp_retention"] = sp_retention
    if sp_coverage.empty:
        tables["sp_coverage_comparison"] = pd.DataFrame()
    else:
        all_trial_coverage = combined_coverage.copy()
        all_trial_coverage.insert(0, "scope", "All trials (primary)")
        conditional_coverage = sp_coverage.copy()
        conditional_coverage.insert(0, "scope", "Decoded S/P only (sensitivity)")
        tables["sp_coverage_comparison"] = pd.concat(
            [all_trial_coverage, conditional_coverage], ignore_index=True
        )
    if sp_classification.empty:
        tables["sp_classification"] = sp_classification
    else:
        classification = sp_classification.groupby(
            ["model", "generating_state", "decoded_state"], as_index=False
        )["n"].sum()
        classification["generating_state_n"] = classification.groupby(
            ["model", "generating_state"]
        )["n"].transform("sum")
        classification["fraction_within_generating_state"] = (
            classification["n"] / classification["generating_state_n"]
        )
        tables["sp_classification"] = classification

    restarts, restart_summary = _restart_summary(out_dir)
    tables["restart_diagnostics"] = restarts
    tables["restart_summary"] = restart_summary
    tables["model_info"] = _read_csv(out_dir / "model_info_criteria.csv")
    tables.update(build_teaching_tables(data, tables))

    for name, frame in tables.items():
        _write_table(frame, report_data_dir, f"{name}.csv")
    return tables


def publication_status(out_dir: str | Path, tables: dict[str, pd.DataFrame]) -> tuple[str, list[str]]:
    out_dir = Path(out_dir)
    issues: list[str] = []
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists():
        issues.append("No authoritative run manifest is present; these outputs are for quality assurance only.")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            issues.append(f"Run manifest status is {manifest.get('status', 'unknown')}, not complete.")
    cv = _read_csv(out_dir / "cv_results.csv", required=False)
    info = _read_csv(out_dir / "model_info_criteria.csv", required=False)
    subject = _read_csv(out_dir / "subject_level_hmm.csv", required=False)
    for label, frame in [("cross-validation", cv), ("final models", info), ("subject fits", subject)]:
        if frame.empty:
            issues.append(f"Required {label} results are missing.")
        elif "converged" in frame and not frame["converged"].fillna(False).astype(bool).all():
            issues.append(f"At least one selected {label.replace('-', ' ')} fit is not marked converged.")
    return ("ready" if not issues else "partial"), issues


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def result_claims(tables: dict[str, pd.DataFrame], status: str) -> dict[str, str]:
    cv = tables["cv_summary"]
    bootstrap = tables["bootstrap"]
    subject = tables["subject"]
    best = cv.sort_values("mean_test_ll_per_trial", ascending=False).iloc[0]
    hmm = bootstrap[bootstrap["model"] == "HMM_static"]
    if hmm.empty:
        hmm_sentence = "The static HMM comparison is unavailable."
    else:
        row = hmm.iloc[0]
        if row["ci_low"] > 0:
            verb = "outperformed"
        elif row["ci_high"] < 0:
            verb = "underperformed"
        else:
            verb = "did not clearly differ from"
        hmm_sentence = (
            f"The static HMM {verb} independent Switching by {_fmt(row['observed_delta_ll_per_trial'], 4)} "
            f"held out log likelihood units per trial (95% sequence bootstrap CI {_fmt(row['ci_low'], 4)} to {_fmt(row['ci_high'], 4)})."
        )
    sensory = subject[subject["state"] == "Sensory"]
    prior = subject[subject["state"] == "Prior"]
    persistence_sentence = (
        f"Across {subject['subject_id'].nunique()} subjects, mean self transition probabilities were "
        f"{_fmt(sensory['self_transition'].mean())} for sensory reliance and {_fmt(prior['self_transition'].mean())} for prior reliance."
    )
    status_sentence = (
        "This report contains publication-run estimates."
        if status == "ready"
        else "This quality assurance rendering is not for publication; incomplete or unconverged outputs must not be cited as final estimates."
    )
    return {
        "status": status_sentence,
        "best_model": f"{best['model_label']} had the highest mean held out log likelihood per trial ({_fmt(best['mean_test_ll_per_trial'], 4)}).",
        "hmm": hmm_sentence,
        "persistence": persistence_sentence,
    }


def _set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ["png", "svg"]:
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def _experiment_figure(figure_dir: Path, summary: pd.DataFrame) -> list[Path]:
    row = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    phases = [
        (0.2, "Fixation", "About 1,000 ms"),
        (2.25, "Motion stimulus", "300 ms; 6%, 12%, or 24%"),
        (4.3, "Direction estimate", "Rotate response line"),
        (6.35, "Confirmation", "Confirm estimate"),
        (8.4, "Feedback", "True direction shown"),
    ]
    for i, (x, title, subtitle) in enumerate(phases):
        rect = plt.Rectangle((x, 1.2), 1.55, 1.05, facecolor=LIGHT_BLUE if i % 2 == 0 else "#F7E3C1", edgecolor=INK, lw=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.775, 1.83, title, ha="center", va="center", fontweight="bold", fontsize=9)
        ax.text(x + 0.775, 1.48, subtitle, ha="center", va="center", fontsize=7, color=MUTED)
        if i < len(phases) - 1:
            ax.annotate("", xy=(phases[i + 1][0] - 0.12, 1.72), xytext=(x + 1.67, 1.72), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.1})
    ax.text(0.3, 2.85, "Motion-direction estimation task and analyzed hierarchy", fontsize=13, fontweight="bold")
    ax.text(
        0.3,
        0.55,
        f"{int(row['subjects'])} subjects | {int(row['usable_trials']):,} usable trials | {int(row['sequences'])} run-level sequences | "
        "4 prior widths (10, 20, 40, 80 deg)",
        fontsize=10,
    )
    ax.text(0.3, 0.2, "Task structure redrawn from the experimental description; no original-paper artwork is reproduced.", fontsize=8, color=MUTED)
    return _save_figure(fig, figure_dir, "figure_00_experiment_design")


def _behavior_histogram_figure(figure_dir: Path, hist: pd.DataFrame, reference: str, suffix: str) -> list[Path]:
    subset = hist[hist["reference"] == reference]
    coherences = sorted(subset["motion_coherence"].unique())
    priors = sorted(subset["prior_std"].unique())
    fig, axes = plt.subplots(len(coherences), len(priors), figsize=(12, 7.4), sharex=True, sharey=True, constrained_layout=True)
    for i, coherence in enumerate(coherences):
        for j, prior_std in enumerate(priors):
            ax = axes[i, j]
            cell = subset[(subset["motion_coherence"] == coherence) & (subset["prior_std"] == prior_std)]
            ax.plot(cell["bin_center_deg"], cell["proportion"], color=BLUE, lw=1.6)
            ax.axvline(0, color=INK, lw=0.8, ls="--")
            ax.grid(axis="y", color=GRID, lw=0.6)
            if i == 0:
                ax.set_title(f"Prior SD {prior_std:g} deg")
            if j == 0:
                ax.set_ylabel(f"Coherence {coherence * 100:g}%\nProportion")
            if i == len(coherences) - 1:
                ax.set_xlabel(f"Response - {reference} (deg)")
    fig.suptitle(f"Observed response distributions relative to the {reference}", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, f"figure_01{suffix}_behavior_relative_{reference}")


def _model_comparison_figure(figure_dir: Path, cv: pd.DataFrame, bootstrap: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ordered = cv.sort_values("mean_test_ll_per_trial")
    axes[0].barh(ordered["model_label"], ordered["mean_test_ll_per_trial"], xerr=ordered["se_test_ll_per_trial"], color=BLUE, alpha=0.9)
    axes[0].set_xlabel("Held-out log likelihood per trial")
    axes[0].set_title("Absolute predictive performance")
    axes[0].grid(axis="x", color=GRID, lw=0.6)
    _panel_label(axes[0], "A")
    delta = bootstrap.sort_values("observed_delta_ll_per_trial")
    xerr = np.vstack([
        delta["observed_delta_ll_per_trial"] - delta["ci_low"],
        delta["ci_high"] - delta["observed_delta_ll_per_trial"],
    ])
    axes[1].errorbar(delta["observed_delta_ll_per_trial"], delta["model_label"], xerr=xerr, fmt="o", color=ORANGE, ecolor=INK, capsize=3)
    axes[1].axvline(0, color=INK, lw=0.9)
    axes[1].set_xlabel("Difference vs independent Switching (LL/trial)")
    axes[1].set_title("Paired sequence-bootstrap differences")
    axes[1].grid(axis="x", color=GRID, lw=0.6)
    _panel_label(axes[1], "B")
    fig.suptitle("Cross-validated model comparison", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_02_cross_validated_model_comparison")


def _hmm_parameter_figure(figure_dir: Path, transition: pd.DataFrame, emissions: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    matrix = transition.pivot(index="previous_state", columns="next_state", values="probability").reindex(index=STATE_NAMES, columns=STATE_NAMES)
    im = axes[0].imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    axes[0].set_xticks(range(3), ["Sensory", "Prior", "Lapse"], rotation=30, ha="right")
    axes[0].set_yticks(range(3), ["Sensory", "Prior", "Lapse"])
    axes[0].set_xlabel("Next state")
    axes[0].set_ylabel("Previous state")
    axes[0].set_title("Transition probabilities")
    for i in range(3):
        for j in range(3):
            value = matrix.iloc[i, j]
            axes[0].text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > 0.55 else INK)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    _panel_label(axes[0], "A")
    sensory = emissions[emissions["parameter"] == "sensory_kappa"]
    axes[1].plot(range(len(sensory)), sensory["value"], marker="o", color=BLUE)
    axes[1].set_xticks(range(len(sensory)), [x.replace("Coherence ", "") for x in sensory["condition_label"]])
    axes[1].set_xlabel("Motion coherence")
    axes[1].set_ylabel("von Mises concentration")
    axes[1].set_title("Sensory-state precision")
    axes[1].grid(axis="y", color=GRID, lw=0.6)
    _panel_label(axes[1], "B")
    prior = emissions[emissions["parameter"] == "prior_kappa"]
    axes[2].plot(range(len(prior)), prior["value"], marker="o", color=ORANGE)
    axes[2].set_xticks(range(len(prior)), [x.replace("Prior SD ", "") for x in prior["condition_label"]])
    axes[2].set_xlabel("Prior standard deviation (deg)")
    axes[2].set_ylabel("von Mises concentration")
    axes[2].set_title("Prior-state precision")
    axes[2].grid(axis="y", color=GRID, lw=0.6)
    _panel_label(axes[2], "C")
    fig.suptitle("Static HMM transition and emission parameters", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_03_hmm_transition_and_emissions")


def _occupancy_figure(figure_dir: Path, occupancy: pd.DataFrame, sequence: pd.DataFrame) -> list[Path]:
    bins = ["near (<30 deg)", "medium (30-60 deg)", "far (>=60 deg)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for panel, conflict_bin in enumerate(bins):
        ax = axes.flat[panel]
        cell = occupancy[occupancy["conflict_bin"] == conflict_bin]
        pivot = cell.pivot(index="motion_coherence", columns="prior_std", values="p_prior")
        im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlOrBr", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{100*x:g}%" for x in pivot.index])
        ax.set_xlabel("Prior SD (deg)")
        ax.set_ylabel("Coherence")
        ax.set_title(f"Prior-state occupancy: {conflict_bin}")
        _panel_label(ax, chr(ord("A") + panel))
    fig.colorbar(im, ax=axes.flat[:3].tolist(), fraction=0.025, pad=0.02, label="Mean posterior P(prior state)")
    ax = axes.flat[3]
    if not sequence.empty:
        x = np.arange(1, len(sequence) + 1)
        ax.plot(x, sequence["p_sensory"], label="Sensory", color=BLUE)
        ax.plot(x, sequence["p_prior"], label="Prior", color=ORANGE)
        ax.plot(x, sequence["p_lapse"], label="Lapse", color=MUTED, ls="--")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial within representative sequence")
        ax.set_ylabel("Posterior state probability")
        ax.legend(loc="upper right")
    ax.set_title("Representative posterior state sequence")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _panel_label(ax, "D")
    fig.suptitle("Posterior state occupancy and temporal organization", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_04_posterior_state_occupancy")


def _subject_figure(figure_dir: Path, subject: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for state, color, marker in [("Sensory", BLUE, "o"), ("Prior", ORANGE, "s")]:
        cell = subject[subject["state"] == state].sort_values("subject_id")
        axes[0].scatter(cell["subject_id"], cell["self_transition"], label=state, color=color, marker=marker)
        axes[0].scatter(cell["subject_id"], cell["stationary_probability"], color=color, marker=marker, facecolors="none", alpha=0.7)
        axes[1].scatter(cell["subject_id"], cell["persistence_excess"], label=state, color=color, marker=marker)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Subject")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Self-transition (filled) and stationary baseline (open)")
    axes[0].legend()
    axes[0].grid(axis="y", color=GRID, lw=0.6)
    _panel_label(axes[0], "A")
    axes[1].axhline(0, color=INK, lw=0.9)
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("Self-transition minus stationary probability")
    axes[1].set_title("Persistence above independent-state expectation")
    axes[1].grid(axis="y", color=GRID, lw=0.6)
    _panel_label(axes[1], "B")
    fig.suptitle("Subject-level sensory and prior persistence", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_05_subject_level_persistence")


def _serial_figure(figure_dir: Path, serial: pd.DataFrame, bootstrap: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    serial_delta = bootstrap[bootstrap["model"].str.startswith("Serial_", na=False)].sort_values("observed_delta_ll_per_trial")
    if not serial_delta.empty:
        xerr = np.vstack([
            serial_delta["observed_delta_ll_per_trial"] - serial_delta["ci_low"],
            serial_delta["ci_high"] - serial_delta["observed_delta_ll_per_trial"],
        ])
        axes[0].errorbar(serial_delta["observed_delta_ll_per_trial"], serial_delta["model_label"], xerr=xerr, fmt="o", color=BLUE, ecolor=INK, capsize=3)
    axes[0].axvline(0, color=INK, lw=0.9)
    axes[0].set_xlabel("Difference vs independent Switching (LL/trial)")
    axes[0].set_title("Held-out serial-control performance")
    axes[0].grid(axis="x", color=GRID, lw=0.6)
    _panel_label(axes[0], "A")
    if not serial.empty:
        x = np.arange(len(serial))
        axes[1].scatter(x, serial["alpha_stim"].fillna(0), color=BLUE, label="Previous stimulus")
        axes[1].scatter(x, serial["alpha_resp"].fillna(0), color=ORANGE, marker="s", label="Previous response")
        axes[1].set_xticks(x, [f"{label}\nfold {fold}" for label, fold in zip(serial["model_label"], serial["fold"])], rotation=45, ha="right", fontsize=7)
    axes[1].axhline(0, color=INK, lw=0.9)
    axes[1].set_ylabel("Serial attraction coefficient")
    axes[1].set_title("Fold-level serial coefficients")
    axes[1].legend()
    axes[1].grid(axis="y", color=GRID, lw=0.6)
    _panel_label(axes[1], "B")
    fig.suptitle("Serial-dependence controls", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_06_serial_dependence_controls")


def _covariate_figure(figure_dir: Path, covariate: pd.DataFrame) -> list[Path]:
    if covariate.empty:
        return []
    stay = covariate[covariate["is_stay"]].sort_values("delta_plus_minus").copy()
    height = max(5.0, 0.34 * len(stay))
    fig, ax = plt.subplots(figsize=(10, height), constrained_layout=True)
    if {"ci_low", "ci_high"}.issubset(stay.columns):
        xerr = np.vstack([stay["delta_plus_minus"] - stay["ci_low"], stay["ci_high"] - stay["delta_plus_minus"]])
        ax.errorbar(stay["delta_plus_minus"], stay["effect_label"], xerr=xerr, fmt="o", color=BLUE, ecolor=INK, capsize=2)
    else:
        ax.scatter(stay["delta_plus_minus"], stay["effect_label"], color=BLUE)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlabel("Change in stay probability (+1 SD minus -1 SD)")
    ax.set_title("Covariate-HMM persistence effects")
    ax.grid(axis="x", color=GRID, lw=0.6)
    return _save_figure(fig, figure_dir, "figure_07_covariate_hmm_effects")


def _ppc_metric_figure(figure_dir: Path, metrics: pd.DataFrame) -> list[Path]:
    selected = [("mean_abs_error_deg", "Mean absolute error (deg)"), ("prior_like_rate", "Prior-like response rate")]
    if "model" not in metrics.columns:
        metrics = metrics.assign(model="HMM_static")
    available = list(metrics["model"].drop_duplicates())
    models = [model for model in ["HMM_static", "Covariate_HMM"] if model in available]
    models += [model for model in available if model not in models]
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4.7 * len(models)), squeeze=False, constrained_layout=True)
    panel = 0
    for row_index, model in enumerate(models):
        for column_index, (metric, label) in enumerate(selected):
            ax = axes[row_index, column_index]
            cell = metrics[(metrics["model"] == model) & (metrics["metric"] == metric)].sort_values(["motion_coherence", "prior_std"])
            x = np.arange(len(cell))
            yerr = np.vstack([cell["simulated_mean"] - cell["simulated_ci_low"], cell["simulated_ci_high"] - cell["simulated_mean"]])
            ax.errorbar(x, cell["simulated_mean"], yerr=yerr, fmt="s", color=ORANGE, ecolor=ORANGE, capsize=2, label="Simulated mean and 95% interval")
            ax.scatter(x, cell["observed"], color=BLUE, label="Observed", zorder=3)
            ax.set_xticks(x, cell["condition"], rotation=45, ha="right")
            ax.set_xlabel("Coherence / prior SD")
            ax.set_ylabel(label)
            ax.set_title(f"{MODEL_LABELS.get(model, model)}: {label}")
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.legend(fontsize=8)
            _panel_label(ax, chr(ord("A") + panel))
            panel += 1
    fig.suptitle("Posterior predictive condition checks", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_08_posterior_predictive_metrics")


def _ppc_histogram_figure(
    figure_dir: Path,
    hist: pd.DataFrame,
    model: str,
    reference: str,
    suffix: str,
    *,
    filename_prefix: str = "figure_09",
    analysis_label: str = "",
) -> list[Path]:
    if "model" not in hist.columns:
        hist = hist.assign(model="HMM_static")
    subset = hist[(hist["model"] == model) & (hist["reference"] == reference)]
    if subset.empty:
        return []
    coherences = sorted(subset["motion_coherence"].unique())
    priors = sorted(subset["prior_std"].unique())
    fig, axes = plt.subplots(len(coherences), len(priors), figsize=(12, 7.4), sharex=True, sharey=True, constrained_layout=True)
    for i, coherence in enumerate(coherences):
        for j, prior_std in enumerate(priors):
            ax = axes[i, j]
            cell = subset[(subset["motion_coherence"] == coherence) & (subset["prior_std"] == prior_std)].sort_values("bin_center_deg")
            ax.fill_between(cell["bin_center_deg"], cell["simulated_ci_low"], cell["simulated_ci_high"], color="#F7E3C1", linewidth=0)
            ax.plot(cell["bin_center_deg"], cell["simulated_mean"], color=ORANGE, lw=1.4, label="Simulated")
            ax.plot(cell["bin_center_deg"], cell["observed"], color=BLUE, lw=1.4, label="Observed")
            ax.axvline(0, color=INK, lw=0.7, ls="--")
            ax.grid(axis="y", color=GRID, lw=0.5)
            if i == 0:
                ax.set_title(f"Prior SD {prior_std:g} deg")
            if j == 0:
                ax.set_ylabel(f"Coherence {100*coherence:g}%\nProportion")
            if i == len(coherences) - 1:
                ax.set_xlabel(f"Response - {reference} (deg)")
    axes[0, -1].legend(fontsize=7)
    title_label = f"{analysis_label} " if analysis_label else ""
    fig.suptitle(
        f"{MODEL_LABELS.get(model, model)} {title_label}response distributions relative to the {reference}",
        fontsize=13,
        fontweight="bold",
    )
    return _save_figure(fig, figure_dir, f"{filename_prefix}{suffix}_ppc_relative_{reference}")


def _sp_coverage_figure(figure_dir: Path, coverage: pd.DataFrame) -> list[Path]:
    if coverage.empty:
        return []
    available = list(coverage["model"].drop_duplicates())
    models = [model for model in ["HMM_static", "Covariate_HMM"] if model in available]
    models += [model for model in available if model not in models]
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(7 * len(models), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    scopes = list(coverage["scope"].drop_duplicates())
    colors = [BLUE, ORANGE]
    for panel, model in enumerate(models):
        ax = axes[0, panel]
        cell = coverage[coverage["model"] == model]
        metrics = list(cell["metric"].drop_duplicates())
        x = np.arange(len(metrics), dtype=float)
        width = 0.36
        for scope_index, scope in enumerate(scopes):
            values = (
                cell[cell["scope"] == scope]
                .set_index("metric")
                .reindex(metrics)["coverage_rate"]
                .to_numpy(dtype=float)
            )
            offset = (scope_index - (len(scopes) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, color=colors[scope_index % len(colors)], label=scope)
        ax.set_xticks(x, [metric.replace("_", " ") for metric in metrics], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Condition-cell coverage proportion")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(fontsize=8)
        _panel_label(ax, chr(ord("A") + panel))
    fig.suptitle("Complete-model and decoded S/P-only PPC coverage", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_11_sp_coverage_sensitivity")


def _run_length_figure(figure_dir: Path, calibration: pd.DataFrame) -> list[Path]:
    if "model" not in calibration.columns:
        calibration = calibration.assign(model="HMM_static")
    mean = calibration[calibration["metric"] == "mean"].copy()
    available = list(mean["model"].drop_duplicates())
    models = [model for model in ["HMM_static", "Covariate_HMM"] if model in available]
    models += [model for model in available if model not in models]
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4.6), squeeze=False, constrained_layout=True)
    for panel, model in enumerate(models):
        ax = axes[0, panel]
        cell = mean[mean["model"] == model]
        x = np.arange(len(cell))
        yerr = np.vstack([cell["simulated_mean"] - cell["simulated_ci_low"], cell["simulated_ci_high"] - cell["simulated_mean"]])
        ax.errorbar(x + 0.1, cell["simulated_mean"], yerr=yerr, fmt="s", color=ORANGE, capsize=3, label="Simulated mean and 95% interval")
        ax.scatter(x - 0.1, cell["observed"], color=BLUE, label="Observed marginal-MAP runs")
        ax.set_xticks(x, cell["state"].str.replace("_", " "))
        ax.set_ylabel("Mean state run length (trials)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(fontsize=8)
        _panel_label(ax, chr(ord("A") + panel))
    fig.suptitle("Observed and simulated state persistence", fontsize=13, fontweight="bold")
    return _save_figure(fig, figure_dir, "figure_10_state_run_length_calibration")


def generate_publication_figures(out_dir: str | Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    _set_publication_style()
    figure_dir = Path(out_dir) / "report" / "figures"
    made: list[Path] = []
    made += _experiment_figure(figure_dir, tables["dataset_summary"])
    made += _behavior_histogram_figure(figure_dir, tables["observed_histograms"], "stimulus", "a")
    made += _behavior_histogram_figure(figure_dir, tables["observed_histograms"], "prior", "b")
    made += _model_comparison_figure(figure_dir, tables["cv_summary"], tables["bootstrap"])
    made += _hmm_parameter_figure(figure_dir, tables["transition"], tables["emissions"])
    made += _occupancy_figure(figure_dir, tables["occupancy"], tables["representative_sequence"])
    made += _subject_figure(figure_dir, tables["subject"])
    made += _serial_figure(figure_dir, tables["serial"], tables["bootstrap"])
    made += _covariate_figure(figure_dir, tables["covariate"])
    made += _ppc_metric_figure(figure_dir, tables["ppc_metrics"])
    made += _ppc_histogram_figure(figure_dir, tables["ppc_histograms"], "HMM_static", "stimulus", "a")
    made += _ppc_histogram_figure(figure_dir, tables["ppc_histograms"], "HMM_static", "prior", "b")
    made += _ppc_histogram_figure(figure_dir, tables["ppc_histograms"], "Covariate_HMM", "stimulus", "c")
    made += _ppc_histogram_figure(figure_dir, tables["ppc_histograms"], "Covariate_HMM", "prior", "d")
    made += _run_length_figure(figure_dir, tables["run_length_calibration"])
    if not tables.get("sp_ppc_metrics", pd.DataFrame()).empty:
        made += _sp_coverage_figure(figure_dir, tables["sp_coverage_comparison"])
        made += _ppc_histogram_figure(
            figure_dir, tables["sp_ppc_histograms"], "HMM_static", "stimulus", "a",
            filename_prefix="figure_12", analysis_label="decoded S/P-only",
        )
        made += _ppc_histogram_figure(
            figure_dir, tables["sp_ppc_histograms"], "HMM_static", "prior", "b",
            filename_prefix="figure_12", analysis_label="decoded S/P-only",
        )
        made += _ppc_histogram_figure(
            figure_dir, tables["sp_ppc_histograms"], "Covariate_HMM", "stimulus", "c",
            filename_prefix="figure_12", analysis_label="decoded S/P-only",
        )
        made += _ppc_histogram_figure(
            figure_dir, tables["sp_ppc_histograms"], "Covariate_HMM", "prior", "d",
            filename_prefix="figure_12", analysis_label="decoded S/P-only",
        )
    return made


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.replace([np.inf, -np.inf], np.nan).to_json(orient="records"))


def _table_source(source_id: str, label: str, path: str) -> dict[str, Any]:
    normalized = path.replace("\\", "/")
    sql_path = normalized.replace("'", "''")
    return {
        "id": source_id,
        "label": label,
        "path": normalized,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{sql_path}')",
            "description": f"Loads the reviewed rows from {label}.",
        },
    }


def _artifact_datasets(tables: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    ppc_mae = tables["ppc_metrics"][tables["ppc_metrics"]["metric"] == "mean_abs_error_deg"]
    ppc_long = []
    for _, row in ppc_mae.iterrows():
        model = str(row.get("model", "HMM_static"))
        label = MODEL_LABELS.get(model, model)
        ppc_long.extend([
            {"condition": row["condition"], "source": f"{label} observed", "model": model, "value": row["observed"], "coherence": row["motion_coherence"], "prior_std": row["prior_std"]},
            {"condition": row["condition"], "source": f"{label} simulated", "model": model, "value": row["simulated_mean"], "coherence": row["motion_coherence"], "prior_std": row["prior_std"]},
        ])
    run_mean = tables["run_length_calibration"][tables["run_length_calibration"]["metric"] == "mean"]
    run_long = []
    for _, row in run_mean.iterrows():
        model = str(row.get("model", "HMM_static"))
        label = MODEL_LABELS.get(model, model)
        run_long.extend([
            {"state": row["state"], "source": f"{label} observed", "model": model, "run_length": row["observed"]},
            {"state": row["state"], "source": f"{label} simulated", "model": model, "run_length": row["simulated_mean"]},
        ])
    cov_stay = tables["covariate"][tables["covariate"].get("is_stay", False)].copy() if not tables["covariate"].empty else pd.DataFrame()
    sp_ppc_long = []
    sp_ppc_mae = tables.get("sp_ppc_metrics", pd.DataFrame())
    if not sp_ppc_mae.empty:
        sp_ppc_mae = sp_ppc_mae[sp_ppc_mae["metric"] == "mean_abs_error_deg"]
        for _, row in sp_ppc_mae.iterrows():
            model = str(row.get("model", "HMM_static"))
            label = MODEL_LABELS.get(model, model)
            sp_ppc_long.extend([
                {"condition": row["condition"], "source": f"{label} observed S/P", "model": model, "value": row["observed"], "coherence": row["motion_coherence"], "prior_std": row["prior_std"]},
                {"condition": row["condition"], "source": f"{label} simulated S/P", "model": model, "value": row["simulated_mean"], "coherence": row["motion_coherence"], "prior_std": row["prior_std"]},
            ])
    datasets = {
        "dataset_summary": _records(tables["dataset_summary"]),
        "behavioral_conditions": _records(tables["behavioral_conditions"]),
        "cv_summary": _records(tables["cv_summary"]),
        "bootstrap": _records(tables["bootstrap"]),
        "transition": _records(tables["transition"]),
        "emissions": _records(tables["emissions"]),
        "subject": _records(tables["subject"]),
        "covariate_stay": _records(cov_stay),
        "ppc_mae": ppc_long,
        "ppc_coverage": _records(tables["ppc_coverage"]),
        "sp_ppc_mae": sp_ppc_long,
        "sp_coverage_comparison": _records(tables.get("sp_coverage_comparison", pd.DataFrame())),
        "sp_retention": _records(tables.get("sp_retention", pd.DataFrame())),
        "sp_classification": _records(tables.get("sp_classification", pd.DataFrame())),
        "run_length": run_long,
        "restart_summary": _records(tables["restart_summary"]),
        "model_info": _records(tables["model_info"]),
        "data_dictionary": _records(tables["data_dictionary"]),
        "notation_glossary": _records(tables["notation_glossary"]),
        "metric_dictionary": _records(tables["metric_dictionary"]),
        "model_catalog": _records(tables["model_catalog"]),
        "analysis_pipeline": _records(tables["analysis_pipeline"]),
        "figure_guide": _records(tables["figure_guide"]),
        "emission_interpretation": _records(tables["emission_interpretation"]),
    }
    return datasets


def _artifact_sources(out_dir: Path, *, include_sp: bool = False) -> list[dict[str, Any]]:
    prefix = f"outputs/{out_dir.name}"
    sources = [
        _table_source("dataset", "Laquitaine and Gardner direction-estimation dataset", "data/data01_direction4priors.csv"),
        {"id": "paper", "label": "Laquitaine and Gardner (2018), A Switching Observer for Human Perceptual Estimation", "href": "https://doi.org/10.1016/j.neuron.2017.12.011"},
        {"id": "proposal", "label": "Temporal-arbitration reanalysis proposal", "path": "PROPOSAL.md"},
        {"id": "config", "label": "Resolved default publication configuration", "path": "configs/default.yaml"},
        {"id": "implementation", "label": "Perceptual-arbitration model implementation", "path": "src/perceptual_arbitration"},
        _table_source("cv", "Cross-validation model results", f"{prefix}/cv_results.csv"),
        _table_source("bootstrap", "Sequence-bootstrap model differences", f"{prefix}/bootstrap_model_differences.csv"),
        _table_source("hmm", "Final static HMM parameters and posterior states", f"{prefix}/hmm_final_parameters.csv"),
        _table_source("subjects", "Subject-level HMM fits", f"{prefix}/subject_level_hmm.csv"),
        _table_source("covariates", "Covariate-HMM effects", f"{prefix}/covariate_hmm_effect_intervals.csv"),
        _table_source("ppc", "Static and covariate HMM posterior predictive intervals", f"{prefix}/posterior_predictive_model_metric_intervals.csv"),
        _table_source("ppc_coverage", "Posterior predictive condition-metric coverage", f"{prefix}/posterior_predictive_model_coverage.csv"),
        _table_source("convergence", "Restart and convergence diagnostics", f"{prefix}/report/data/restart_summary.csv"),
        _table_source("data_dictionary_source", "Generated data dictionary", f"{prefix}/report/data/data_dictionary.csv"),
        _table_source("notation_source", "Generated notation glossary", f"{prefix}/report/data/notation_glossary.csv"),
        _table_source("metric_source", "Generated metric dictionary", f"{prefix}/report/data/metric_dictionary.csv"),
        _table_source("model_catalog_source", "Generated model catalog", f"{prefix}/report/data/model_catalog.csv"),
        _table_source("pipeline_source", "Generated analysis pipeline", f"{prefix}/report/data/analysis_pipeline.csv"),
        _table_source("emission_interpretation_source", "Generated emission interpretation", f"{prefix}/report/data/emission_interpretation.csv"),
        _table_source("figure_guide_source", "Generated complete visual guide", f"{prefix}/report/data/figure_guide.csv"),
    ]
    if include_sp:
        sources.extend([
            _table_source("sp_ppc", "Decoded S/P-only posterior predictive intervals", f"{prefix}/posterior_predictive_sp_model_metric_intervals.csv"),
            _table_source("sp_coverage", "All-trial versus decoded S/P-only PPC coverage", f"{prefix}/report/data/sp_coverage_comparison.csv"),
            _table_source("sp_retention", "Decoded S/P-only retained-trial summaries", f"{prefix}/posterior_predictive_sp_model_retention_summary.csv"),
            _table_source("sp_classification", "Generating-state versus decoded-state classification", f"{prefix}/report/data/sp_classification.csv"),
        ])
    return sources


def _validate_csv_sources(sources: list[dict[str, Any]]) -> None:
    import duckdb

    project_root = Path(__file__).resolve().parents[2]
    connection = duckdb.connect()
    try:
        for source in sources:
            path = source.get("path")
            if not isinstance(path, str) or not path.lower().endswith(".csv"):
                continue
            absolute = project_root / Path(path)
            if not absolute.exists():
                continue
            connection.execute("SELECT * FROM read_csv_auto(?) LIMIT 1", [str(absolute)]).fetchall()
    finally:
        connection.close()


def _chart_map(figure_guide: pd.DataFrame) -> pd.DataFrame:
    return figure_guide.rename(columns={
        "visual_id": "chart_or_figure_id",
        "axes": "axes_and_layout",
        "marks": "marks_and_colors",
    }).copy()


def sp_coverage_change_summary(
    all_trial_coverage: pd.DataFrame,
    sp_coverage: pd.DataFrame,
) -> str:
    """Describe model-specific S/P sensitivity changes from computed coverage."""
    if sp_coverage.empty:
        return "The decoded S/P-only sensitivity analysis is unavailable."
    all_totals = all_trial_coverage.groupby("model", as_index=False).agg(
        cells=("cells", "sum"), covered=("covered", "sum")
    )
    sp_totals = sp_coverage.groupby("model", as_index=False).agg(
        cells=("cells", "sum"), covered=("covered", "sum")
    )
    all_lookup = {
        str(row.model): (int(row.covered), int(row.cells))
        for row in all_totals.itertuples(index=False)
    }
    clauses = []
    for row in sp_totals.itertuples(index=False):
        model = str(row.model)
        before_covered, before_cells = all_lookup.get(model, (0, 0))
        after_covered, after_cells = int(row.covered), int(row.cells)
        if after_covered > before_covered:
            direction = "improved"
        elif after_covered < before_covered:
            direction = "deteriorated"
        else:
            direction = "did not change"
        clauses.append(
            f"{MODEL_LABELS.get(model, model)} coverage {direction} from "
            f"{before_covered}/{before_cells} to {after_covered}/{after_cells} condition-metric cells"
        )
    return "; ".join(clauses) + "."


def build_report_artifact(
    out_dir: str | Path,
    tables: dict[str, pd.DataFrame],
    status: str,
    issues: list[str],
) -> Path:
    out_dir = Path(out_dir)
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    sp_available = not tables.get("sp_ppc_metrics", pd.DataFrame()).empty
    sources = _artifact_sources(out_dir, include_sp=sp_available)
    _validate_csv_sources(sources)
    claims = result_claims(tables, status)
    teaching = teaching_claims(tables)
    summary = tables["dataset_summary"].iloc[0]
    cov = tables["covariate"]
    previous_error = cov[(cov.get("covariate") == "prev_error") & (cov.get("is_stay", False))] if not cov.empty else pd.DataFrame()
    if previous_error.empty:
        covariate_text = "Previous-error transition effects are unavailable and are not used as evidence."
    elif {"ci_low", "ci_high"}.issubset(previous_error.columns) and ((previous_error["ci_low"] <= 0) & (previous_error["ci_high"] >= 0)).any():
        covariate_text = "At least one previous-error persistence interval includes zero; feedback-error effects remain exploratory."
    else:
        covariate_text = "Previous-error effects are reported as conditional associations, not causal feedback effects."

    summary_status = "The completed 25-restart publication run passed the convergence and completeness gate." if status == "ready" else "This output is for quality assurance and is not ready for publication."
    hmm_bootstrap = tables["bootstrap"][tables["bootstrap"]["model"] == "HMM_static"]
    if hmm_bootstrap.empty:
        summary_hmm = "The static HMM comparison is unavailable."
    else:
        row = hmm_bootstrap.iloc[0]
        summary_hmm = f"Static HMM difference: {_fmt(row['observed_delta_ll_per_trial'], 4)} LL per trial; 95% interval {_fmt(row['ci_low'], 4)} to {_fmt(row['ci_high'], 4)}."

    behavior = tables["behavioral_conditions"]
    condition_min = int(behavior["n"].min())
    condition_max = int(behavior["n"].max())
    ppc_coverage = teaching["ppc_coverage"]
    ppc_coverage_text = "; ".join(
        f"{MODEL_LABELS.get(str(row.model), str(row.model))} {row.metric}: {int(row.covered)}/{int(row.cells)}"
        for row in ppc_coverage.itertuples(index=False)
    )
    ppc_model_totals = ppc_coverage.groupby("model", as_index=False).agg(cells=("cells", "sum"), covered=("covered", "sum"))
    ppc_model_text = "; ".join(
        f"{MODEL_LABELS.get(str(row.model), str(row.model))} covered {int(row.covered)}/{int(row.cells)} condition-metric cells"
        for row in ppc_model_totals.itertuples(index=False)
    )
    total_by_model = {str(row.model): (int(row.covered), int(row.cells)) for row in ppc_model_totals.itertuples(index=False)}
    static_covered, static_cells = total_by_model.get("HMM_static", (0, 0))
    covariate_covered, covariate_cells = total_by_model.get("Covariate_HMM", (0, 0))
    if covariate_cells == 0:
        covariate_ppc_comparison = "A covariate-HMM PPC was not available."
    elif covariate_covered > static_covered:
        covariate_ppc_comparison = f"The covariate HMM improved coverage from {static_covered}/{static_cells} to {covariate_covered}/{covariate_cells} selected condition-metric cells."
    elif covariate_covered == static_covered:
        covariate_ppc_comparison = f"The covariate HMM did not improve the total selected-cell coverage ({covariate_covered}/{covariate_cells} versus {static_covered}/{static_cells})."
    else:
        covariate_ppc_comparison = f"The covariate HMM covered fewer selected condition-metric cells ({covariate_covered}/{covariate_cells}) than the static HMM ({static_covered}/{static_cells})."
    n_ppc_simulations = teaching["ppc_simulations"]
    ppc_draw_text = f"{n_ppc_simulations} complete replicated datasets per model" if n_ppc_simulations is not None else "the available complete replicated datasets"
    run_calibration = tables["run_length_calibration"].copy()
    run_calibration["covered"] = (
        (run_calibration["observed"] >= run_calibration["simulated_ci_low"])
        & (run_calibration["observed"] <= run_calibration["simulated_ci_high"])
    )
    run_mean = run_calibration[run_calibration["metric"] == "mean"]
    run_coverage_text = "; ".join(
        f"{MODEL_LABELS.get(str(model), str(model))} {int(group['covered'].sum())}/{len(group)} mean state-run summaries covered"
        for model, group in run_mean.groupby("model")
    )
    sp_change_text = sp_coverage_change_summary(
        ppc_coverage,
        tables.get("sp_ppc_coverage", pd.DataFrame()),
    )
    sp_retention_text = ""
    if sp_available:
        retention_clauses = []
        for model, group in tables["sp_retention"].groupby("model"):
            observed_total = int(group["observed_total_n"].sum())
            observed_retained = int(group["observed_retained_n"].sum())
            simulated_retained = float(group["simulated_retained_n_mean"].sum())
            retention_clauses.append(
                f"{MODEL_LABELS.get(str(model), str(model))} retained "
                f"{observed_retained:,}/{observed_total:,} observed trials "
                f"and a mean {simulated_retained:,.1f}/{observed_total:,} simulated trials"
            )
        sp_retention_text = "; ".join(retention_clauses) + "."

    cards: list[dict[str, Any]] = []
    charts = [
        {"id": "cv_chart", "title": "Held out model performance", "subtitle": "Four fold sequence stratified cross validation; higher is better.", "type": "horizontalBar", "dataset": "cv_summary", "sourceId": "cv", "encodings": {"x": {"field": "model_label", "type": "nominal", "label": "Model"}, "y": {"field": "mean_test_ll_per_trial", "type": "quantitative", "label": "Mean held out LL/trial"}}},
        {"id": "delta_chart", "title": "Model differences versus independent Switching", "subtitle": "Paired sequence-level held-out log-likelihood differences per trial.", "type": "horizontalBar", "dataset": "bootstrap", "sourceId": "bootstrap", "encodings": {"x": {"field": "model_label", "type": "nominal", "label": "Model"}, "y": {"field": "observed_delta_ll_per_trial", "type": "quantitative", "label": "Difference in LL/trial"}}},
        {"id": "transition_chart", "title": "Static HMM transition matrix", "subtitle": "Rows are previous states and columns are next states.", "type": "heatmap", "dataset": "transition", "sourceId": "hmm", "encodings": {"x": {"field": "next_state", "type": "nominal", "label": "Next state"}, "y": {"field": "probability", "type": "quantitative", "label": "Probability"}, "color": {"field": "previous_state", "type": "nominal", "label": "Previous state"}}},
        {"id": "emission_chart", "title": "HMM emission concentration parameters", "subtitle": "Larger von Mises concentration indicates a narrower response distribution.", "type": "bar", "dataset": "emissions", "sourceId": "hmm", "encodings": {"x": {"field": "condition_label", "type": "nominal", "label": "Condition"}, "y": {"field": "value", "type": "quantitative", "label": "Concentration"}, "color": {"field": "family", "type": "nominal", "label": "Emission family"}}},
        {"id": "subject_chart", "title": "Subject-level self-transition probabilities", "subtitle": "Sensory- and prior-state persistence for each participant.", "type": "scatter", "dataset": "subject", "sourceId": "subjects", "encodings": {"x": {"field": "subject_id", "type": "nominal", "label": "Subject"}, "y": {"field": "self_transition", "type": "quantitative", "label": "Self-transition probability"}, "color": {"field": "state", "type": "nominal", "label": "State"}}},
        {"id": "covariate_chart", "title": "Covariate effects on state persistence", "subtitle": "Change in stay probability from -1 SD to +1 SD, conditional on fitted state responsibilities.", "type": "horizontalBar", "dataset": "covariate_stay", "sourceId": "covariates", "encodings": {"x": {"field": "effect_label", "type": "nominal", "label": "Covariate and state"}, "y": {"field": "delta_plus_minus", "type": "quantitative", "label": "Change in stay probability"}}},
        {"id": "ppc_chart", "title": "Observed and simulated mean absolute error", "subtitle": "Static and covariate HMM checks across motion-coherence / prior-width cells.", "type": "line", "dataset": "ppc_mae", "sourceId": "ppc", "encodings": {"x": {"field": "condition", "type": "nominal", "label": "Coherence / prior SD"}, "y": {"field": "value", "type": "quantitative", "label": "Mean absolute error (deg)"}, "color": {"field": "source", "type": "nominal", "label": "Model and source"}}},
        {"id": "run_chart", "title": "Observed and simulated mean state run lengths", "subtitle": "Each HMM is compared with its own forward-backward marginal-MAP state sequence; these are not Viterbi paths.", "type": "bar", "dataset": "run_length", "sourceId": "ppc", "encodings": {"x": {"field": "state", "type": "nominal", "label": "State"}, "y": {"field": "run_length", "type": "quantitative", "label": "Mean run length"}, "color": {"field": "source", "type": "nominal", "label": "Model and source"}}},
    ]
    for chart in charts:
        chart["layout"] = "full"
    if sp_available:
        charts.append({
            "id": "sp_ppc_chart",
            "title": "Decoded S/P-only mean absolute error",
            "subtitle": "The same smoothed marginal-MAP lapse exclusion is applied to observed and simulated sequences.",
            "type": "line",
            "dataset": "sp_ppc_mae",
            "sourceId": "sp_ppc",
            "layout": "full",
            "encodings": {
                "x": {"field": "condition", "type": "nominal", "label": "Coherence / prior SD"},
                "y": {"field": "value", "type": "quantitative", "label": "Mean absolute error (deg)"},
                "color": {"field": "source", "type": "nominal", "label": "Model and source"},
            },
        })
    tables_spec = [
        {"id": "bootstrap_table", "title": "Paired model differences", "subtitle": "Sequence-bootstrap estimates and 95% percentile intervals.", "dataset": "bootstrap", "sourceId": "bootstrap", "defaultSort": {"field": "observed_delta_ll_per_trial", "direction": "desc"}, "columns": [
            {"field": "model_label", "label": "Model", "type": "text"},
            {"field": "observed_delta_ll_per_trial", "label": "Difference LL/trial", "format": "number"},
            {"field": "ci_low", "label": "CI low", "format": "number"},
            {"field": "ci_high", "label": "CI high", "format": "number"},
            {"field": "p_delta_le_0", "label": "Nonpositive tail proportion", "format": "number"},
        ]},
        {"id": "restart_table", "title": "Convergence and restart stability", "subtitle": "Selected fit status and likelihood spread across deterministic restarts.", "dataset": "restart_summary", "sourceId": "convergence", "defaultSort": {"field": "n_restarts", "direction": "desc"}, "columns": [
            {"field": "stage", "label": "Stage", "type": "text"},
            {"field": "model", "label": "Model", "type": "text"},
            {"field": "fold", "label": "Fold", "format": "number"},
            {"field": "subject_id", "label": "Subject", "format": "number"},
            {"field": "n_restarts", "label": "Restarts", "format": "number"},
            {"field": "n_converged", "label": "Converged restarts", "format": "number"},
            {"field": "selected_converged", "label": "Selected converged", "type": "text"},
            {"field": "ll_range", "label": "LL range", "format": "number"},
        ]},
        {"id": "model_info_table", "title": "In-sample information criteria", "subtitle": "Secondary diagnostics; held-out likelihood remains the primary comparison.", "dataset": "model_info", "sourceId": "hmm", "defaultSort": {"field": "bic", "direction": "asc"}, "columns": [
            {"field": "model", "label": "Model", "type": "text"},
            {"field": "n_parameters", "label": "Parameters", "format": "number"},
            {"field": "aic", "label": "AIC", "format": "number"},
            {"field": "bic", "label": "BIC", "format": "number"},
            {"field": "converged", "label": "Converged", "type": "text"},
        ]},
        {"id": "ppc_coverage_table", "title": "Posterior predictive condition-metric coverage", "subtitle": "Observed values inside 2.5th-97.5th simulation percentiles; diagnostic rather than formal pass/fail testing.", "dataset": "ppc_coverage", "sourceId": "ppc_coverage", "defaultSort": {"field": "model", "direction": "asc"}, "columns": [
            {"field": "model", "label": "Model", "type": "text"},
            {"field": "metric", "label": "Metric", "type": "text"},
            {"field": "covered", "label": "Cells covered", "format": "number"},
            {"field": "cells", "label": "Cells tested", "format": "number"},
            {"field": "coverage_rate", "label": "Coverage proportion", "format": "number"},
        ]},
        {"id": "data_dictionary_table", "title": "Data dictionary", "subtitle": "Raw and derived variables used by the reanalysis.", "dataset": "data_dictionary", "sourceId": "data_dictionary_source", "columns": [
            {"field": "field", "label": "Field", "type": "text"},
            {"field": "origin", "label": "Origin", "type": "text"},
            {"field": "meaning", "label": "Meaning", "type": "text"},
            {"field": "unit", "label": "Unit", "type": "text"},
            {"field": "analysis_role", "label": "Analysis role", "type": "text"},
        ]},
        {"id": "notation_table", "title": "Notation glossary", "subtitle": "Symbols used in the mathematical panels.", "dataset": "notation_glossary", "sourceId": "notation_source", "columns": [
            {"field": "symbol", "label": "Symbol", "type": "text"},
            {"field": "meaning", "label": "Meaning", "type": "text"},
            {"field": "unit_or_range", "label": "Unit or range", "type": "text"},
        ]},
        {"id": "metric_dictionary_table", "title": "How model quality is measured", "subtitle": "Definitions, preferred directions, and interpretation limits.", "dataset": "metric_dictionary", "sourceId": "metric_source", "columns": [
            {"field": "metric", "label": "Metric", "type": "text"},
            {"field": "definition", "label": "Definition", "type": "text"},
            {"field": "direction", "label": "How to read it", "type": "text"},
            {"field": "caveat", "label": "Caveat", "type": "text"},
        ]},
        {"id": "model_catalog_table", "title": "Model catalog and run status", "subtitle": "Contextual models are explained but are absent from this run's numerical ranking.", "dataset": "model_catalog", "sourceId": "model_catalog_source", "columns": [
            {"field": "label", "label": "Model", "type": "text"},
            {"field": "temporal_structure", "label": "Temporal structure", "type": "text"},
            {"field": "core_mechanism", "label": "Mechanism", "type": "text"},
            {"field": "fitting_method", "label": "Fitting", "type": "text"},
            {"field": "parameters", "label": "Parameters", "type": "text"},
            {"field": "result_status", "label": "Status", "type": "text"},
        ]},
        {"id": "analysis_pipeline_table", "title": "Analysis pipeline", "subtitle": "Every stage from CSV ingestion to the fit-independent report renderer.", "dataset": "analysis_pipeline", "sourceId": "pipeline_source", "defaultSort": {"field": "step", "direction": "asc"}, "columns": [
            {"field": "step", "label": "Step", "format": "number"},
            {"field": "stage", "label": "Stage", "type": "text"},
            {"field": "input", "label": "Input", "type": "text"},
            {"field": "operation", "label": "Operation", "type": "text"},
            {"field": "output", "label": "Output", "type": "text"},
        ]},
        {"id": "emission_interpretation_table", "title": "Emission concentration translated to circular spread", "subtitle": "Circular SD is computed from each fitted von Mises concentration.", "dataset": "emission_interpretation", "sourceId": "emission_interpretation_source", "columns": [
            {"field": "family", "label": "State family", "type": "text"},
            {"field": "condition_label", "label": "Condition", "type": "text"},
            {"field": "value", "label": "Kappa", "format": "number"},
            {"field": "circular_sd_deg", "label": "Circular SD (degrees)", "format": "number"},
            {"field": "plain_interpretation", "label": "Plain-language reading", "type": "text"},
        ]},
        {"id": "figure_guide_table", "title": "Complete visual guide", "subtitle": "Question, encoding, numerical takeaway, and caveat for every graph.", "dataset": "figure_guide", "sourceId": "figure_guide_source", "columns": [
            {"field": "title", "label": "Visual", "type": "text"},
            {"field": "question", "label": "Question", "type": "text"},
            {"field": "takeaway", "label": "Takeaway", "type": "text"},
            {"field": "caveat", "label": "Caveat", "type": "text"},
        ]},
    ]
    if sp_available:
        tables_spec.extend([
            {
                "id": "sp_coverage_table",
                "title": "Complete-model versus decoded S/P-only coverage",
                "subtitle": "The complete-model PPC is primary; the decoded S/P-only rows are a conditional sensitivity analysis.",
                "dataset": "sp_coverage_comparison",
                "sourceId": "sp_coverage",
                "defaultSort": {"field": "model", "direction": "asc"},
                "columns": [
                    {"field": "scope", "label": "Analysis scope", "type": "text"},
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "metric", "label": "Metric", "type": "text"},
                    {"field": "covered", "label": "Cells covered", "format": "number"},
                    {"field": "cells", "label": "Cells tested", "format": "number"},
                    {"field": "coverage_rate", "label": "Coverage proportion", "format": "number"},
                ],
            },
            {
                "id": "sp_retention_table",
                "title": "Decoded S/P-only retention by condition",
                "subtitle": "Observed counts use each model's own smoothed marginal-MAP states; simulation columns summarize the same rule over replicated datasets.",
                "dataset": "sp_retention",
                "sourceId": "sp_retention",
                "columns": [
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "condition", "label": "Coherence / prior SD", "type": "text"},
                    {"field": "observed_total_n", "label": "Observed total", "format": "number"},
                    {"field": "observed_retained_n", "label": "Observed retained", "format": "number"},
                    {"field": "observed_retained_rate", "label": "Observed retained proportion", "format": "number"},
                    {"field": "simulated_retained_n_mean", "label": "Mean simulated retained", "format": "number"},
                    {"field": "simulated_retained_rate_ci_low", "label": "Simulated rate CI low", "format": "number"},
                    {"field": "simulated_retained_rate_ci_high", "label": "Simulated rate CI high", "format": "number"},
                ],
            },
            {
                "id": "sp_classification_table",
                "title": "Generating state versus smoothed marginal-MAP decoded state",
                "subtitle": "Simulation-only audit table; the generating state is not used to decide exclusion.",
                "dataset": "sp_classification",
                "sourceId": "sp_classification",
                "columns": [
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "generating_state", "label": "Generating state", "type": "text"},
                    {"field": "decoded_state", "label": "Decoded state", "type": "text"},
                    {"field": "n", "label": "Trials across simulations", "format": "number"},
                    {"field": "fraction_within_generating_state", "label": "Fraction within generating state", "format": "number"},
                ],
            },
        ])

    guide = tables["figure_guide"]
    figure_dir = report_dir / "figures"
    math_panels = model_math_panels()

    def guide_block(visual_id: str) -> dict[str, Any]:
        return {
            "id": f"guide_{visual_id}",
            "type": "markdown",
            "sourceId": "implementation",
            "body": visual_explanation(guide, visual_id),
        }

    def svg_block(block_id: str, filename: str, title: str, alt: str, caption: str) -> dict[str, Any]:
        figure_path = figure_dir / filename
        if not figure_path.exists():
            return {
                "id": block_id,
                "type": "markdown",
                "sourceId": "implementation",
                "body": f"### {title}\n\nThis figure is unavailable because the corresponding model-specific PPC output has not been generated in this output directory.",
            }
        return {
            "id": block_id,
            "type": "html",
            "sourceId": "paper" if block_id == "task_figure" else "implementation",
            "body": embedded_svg_figure(figure_path, title, alt, caption),
        }

    sp_blocks: list[dict[str, Any]] = []
    if sp_available:
        sp_blocks = [
            {
                "id": "sp_sensitivity_section",
                "type": "markdown",
                "sourceId": "sp_ppc",
                "body": (
                    "## Sensitivity check: decoded sensory/prior trials only\n\n"
                    "The lapse emission is circular uniform and remains a required part of each complete generative model. "
                    "Accordingly, the all-trial PPC above remains the primary absolute-adequacy check. This supplementary "
                    "analysis asks a narrower question: after fitting and simulation are complete, do response summaries "
                    "look better calibrated among trials classified as sensory or prior?\n\n"
                    "For every observed and simulated run, forward-backward smoothing was recomputed and each trial was "
                    "assigned the state with the largest smoothed marginal probability. Trials assigned `L_lapse` were "
                    "removed from condition metrics and histograms. This is the same symmetric rule on both sides. It is "
                    "**not Viterbi decoding**, does not produce a single most-probable joint path, uses the complete response "
                    "sequence, and therefore cannot serve as an online lapse detector. For covariate-HMM simulations, "
                    "previous error was rebuilt recursively from simulated responses and standardized with the final "
                    "all-data scaler; observed responses never entered simulated decoding.\n\n"
                    f"**Computed result.** {sp_change_text} {sp_retention_text} Removing difficult lapse-classified trials "
                    "does not automatically make either model adequate. Held-out predictive likelihood, complete-model PPC "
                    "adequacy, and conditional S/P-only adequacy remain three distinct criteria."
                ),
            },
            guide_block("sp_coverage_figure"),
            svg_block(
                "sp_coverage_figure",
                "figure_11_sp_coverage_sensitivity.svg",
                "Figure 12. Complete-model versus decoded S/P-only coverage",
                "Coverage proportions for all-trial and decoded sensory/prior-only posterior predictive checks",
                "The complete-model bars remain authoritative; conditional bars use symmetric smoothed marginal-MAP exclusion.",
            ),
            guide_block("sp_ppc_chart"),
            {"id": "sp_ppc_block", "type": "chart", "chartId": "sp_ppc_chart"},
            {"id": "sp_coverage_table_block", "type": "table", "tableId": "sp_coverage_table"},
            {"id": "sp_retention_table_block", "type": "table", "tableId": "sp_retention_table"},
            {"id": "sp_classification_table_block", "type": "table", "tableId": "sp_classification_table"},
            guide_block("sp_static_stimulus_figure"),
            svg_block(
                "sp_static_stimulus_figure",
                "figure_12a_ppc_relative_stimulus.svg",
                "Figure 13a. Static-HMM decoded S/P response shapes relative to stimulus",
                "Observed and static-HMM simulated response histograms after symmetric decoded-lapse exclusion",
                "Observed and simulated distributions use model-specific smoothed marginal-MAP S/P classifications.",
            ),
            guide_block("sp_static_prior_figure"),
            svg_block(
                "sp_static_prior_figure",
                "figure_12b_ppc_relative_prior.svg",
                "Figure 13b. Static-HMM decoded S/P response shapes relative to prior",
                "Observed and static-HMM simulated response histograms relative to prior after decoded-lapse exclusion",
                "Removing a lapse-classified trial does not merge runs and does not modify the primary run-length PPC.",
            ),
            guide_block("sp_covariate_stimulus_figure"),
            svg_block(
                "sp_covariate_stimulus_figure",
                "figure_12c_ppc_relative_stimulus.svg",
                "Figure 13c. Covariate-HMM decoded S/P response shapes relative to stimulus",
                "Observed and covariate-HMM simulated response histograms after symmetric decoded-lapse exclusion",
                "Simulated decoding reconstructs previous error from simulated responses without observed-response leakage.",
            ),
            guide_block("sp_covariate_prior_figure"),
            svg_block(
                "sp_covariate_prior_figure",
                "figure_12d_ppc_relative_prior.svg",
                "Figure 13d. Covariate-HMM decoded S/P response shapes relative to prior",
                "Observed and covariate-HMM simulated response histograms relative to prior after decoded-lapse exclusion",
                "This is a post-hoc conditional diagnostic, not a new fitted model or a causal state classification.",
            ),
        ]

    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": "# Temporally Persistent Strategy Arbitration in Human Perceptual Inference\n\nA layered teaching and technical report on the four-prior motion-direction experiment."},
        {"id": "how_to_read", "type": "markdown", "body": "## How to read this study\n\nStart with the one-minute explanation, then use each figure's **Question / How to read / Takeaway / Caveat** box. Terms and metrics are defined before they are used. Technical readers can open the collapsed mathematical derivations; no mathematics is required to follow the visible result narrative.\n\n**Evidence labels.** **Original-paper finding** describes Laquitaine and Gardner's published interpretation. **This reanalysis** describes calculations performed here. **Mathematical interpretation** explains what a parameter means under a model. **Supported conclusion** is limited to results with held-out or uncertainty evidence. **Unresolved limitation** marks what these data cannot establish."},
        {"id": "contents", "type": "markdown", "body": "## Contents\n\n[One-minute explanation](#one_minute) | [Experiment](#experiment_section) | [Data pipeline](#data_section) | [Models](#model_roadmap) | [Model evaluation](#evaluation_section) | [Results](#model_results) | [Posterior checks](#ppc_section) | [S/P sensitivity](#sp_sensitivity_section) | [Limitations](#limitations_section) | [Glossary](#notation_section)"},
        {"id": "one_minute", "type": "markdown", "body": f"## One-minute explanation\n\nPeople viewed a noisy field of moving dots and reported its direction. Directions were drawn around a stable learned center, so a response could follow the current visual evidence or the learned prior. The original paper showed that full response distributions can contain both sensory-centered and prior-centered peaks, motivating a Switching Observer.\n\n**This reanalysis asks a new question:** are those response strategies chosen independently on every trial, or do they persist in runs? The best held-out predictor was the {teaching['best_model']} at {_fmt(teaching['best_ll'], 4)} LL/trial. Relative to independent Switching, the static HMM improved predictive density by a geometric-average factor of {teaching['static_density_ratio']:.3f} per response, and the covariate HMM by {teaching['covariate_density_ratio']:.3f}. Sensory and prior self-transition probabilities were {teaching['A_SS']:.3f} and {teaching['A_PP']:.3f}.\n\n**Critical qualification:** winning a relative prediction comparison does not mean a model is absolutely adequate. Across {ppc_draw_text}, {ppc_model_text}. {covariate_ppc_comparison}"},
        {"id": "technical_summary", "type": "markdown", "body": f"## Technical summary\n\n**Status.** {summary_status}\n\n**Prediction.** {claims['best_model']}\n\n**Static HMM.** {summary_hmm}\n\n**Persistence.** {claims['persistence']}\n\n**Adequacy.** PPC coverage by metric was {ppc_coverage_text}. Relative predictive superiority and absolute model adequacy are separate questions."},
        {"id": "provenance", "type": "markdown", "body": "## What comes from where\n\n**Original paper.** The task, behavioral motivation, Basic Bayesian observer, and condition-dependent Switching observer are attributed to Laquitaine and Gardner (2018).\n\n**This reanalysis.** Circular preprocessing, run-sequence construction, independent and serial baselines, static/covariate HMMs, held-out comparisons, bootstrap intervals, subject fits, and posterior predictive checks were produced by this package.\n\n**Context-only models.** Basic Bayesian and original condition-dependent Switching are fully explained below but were not refitted and do not appear in the authoritative numerical ranking.\n\n**Evidence scope.** All numerical claims come from `outputs/full_run`; this renderer reads final tables and never refits a model."},
        {"id": "experiment_section", "type": "markdown", "body": f"## Experimental design: what participants actually did\n\n**Original-paper procedure.** Twelve participants completed at least five computerized sessions. A session contained about four to five blocks, and a block about 200 trials. On each trial, participants fixated for about 1 second, viewed a random-dot motion stimulus for about 300 ms, rotated a line to report perceived direction, confirmed the response, and saw the true direction as feedback. Motion coherence varied trial by trial at 6%, 12%, or 24%; larger coherence supplied more reliable visual motion.\n\nEach block drew directions around a fixed 225-degree center with one of four widths: 10, 20, 40, or 80 degrees. The apparatus therefore combined a random-dot display with an adjustable direction-report line and trial feedback. Participants were instructed to respond accurately and quickly, but were not told the Bayesian or Switching hypothesis and had no separate explicit prior-training phase. Instead, repeated directions and feedback let them learn the stable center and adapt to block-specific width implicitly.\n\nThe analyzed hierarchy contains {int(summary['sessions'])} sessions and {int(summary['sequences'])} run/block sequences. Runs, rather than arbitrary stretches of rows, are the temporal units."},
        guide_block("task_figure"),
        svg_block("task_figure", "figure_00_experiment_design.svg", "Figure 1. Experiment and analysis hierarchy", "Schematic of fixation, random-dot motion, direction report, confirmation, feedback, and the subject-session-run-trial hierarchy", "The figure redraws the experimental sequence from the paper's methods and adds counts from the analyzed dataset."),
        {"id": "data_section", "type": "markdown", "body": f"## From CSV coordinates to analysis-ready sequences\n\nThe raw file has {int(summary['raw_rows']):,} rows. The loader requires the two response-vector coordinates and the experimental condition fields; it dropped exactly {int(summary['dropped_rows'])} rows with missing response coordinates, leaving {int(summary['usable_trials']):,} trials. Response angle is recovered from those coordinates with the two-argument arctangent and wrapped from 0 inclusive to 360 exclusive degrees. Every difference between two angles uses the shortest signed arc, wrapped from minus 180 inclusive to 180 exclusive degrees, so 359 and 1 degrees differ by 2 rather than 358 degrees.\n\nRows are sorted by subject, session, run, and trial index. Each subject-session-run group becomes one sequence; previous-trial variables are reset at its first trial. This prevents a previous response from the end of one block leaking into the next. The 12 coherence-by-prior-width cells contain {condition_min:,} to {condition_max:,} trials.\n\nFor covariate HMM cross-validation, means and standard deviations are learned on training sequences only, then reused on the held-out fold. Thus the contrast between plus one and minus one training standard deviation is defined by the training data and does not inspect held-out outcomes."},
        {"id": "pipeline_table_block", "type": "table", "tableId": "analysis_pipeline_table"},
        {"id": "data_dictionary_block", "type": "table", "tableId": "data_dictionary_table"},
        {"id": "circular_section", "type": "markdown", "body": "### Why circular statistics are necessary\n\nDirections live on a circle, so ordinary subtraction and Gaussian error can be wrong near zero degrees. The analysis uses wrapped differences, circular means, resultant lengths, and von Mises densities. The von Mises distribution is the circular analogue of a normal density: its center is an angle and its concentration controls width."},
        {"id": "circular_math", "type": "html", "sourceId": "implementation", "body": math_panels["circular_math"]},
        {"id": "behavior_section", "type": "markdown", "body": "## Behavioral replication and response distributions\n\n**Original-paper finding.** Mean and variance can look compatible with Bayesian integration even when the full distribution is bimodal, with one peak near the current stimulus and another near the learned prior.\n\n**This reanalysis.** For each of 12 condition cells, the report computes signed and absolute circular error, circular spread, cosine accuracy, the fraction of responses closer to the prior than the stimulus, and histograms aligned separately to stimulus and prior. Higher coherence narrows stimulus-centered responses. Narrower priors increase prior-centered structure. These are descriptive observations; a prior-like trial is not automatically a recovered prior state."},
        guide_block("behavior_stimulus_figure"),
        svg_block("behavior_stimulus_figure", "figure_01a_behavior_relative_stimulus.svg", "Figure 2a. Responses aligned to the stimulus", "Twelve histograms of response minus stimulus, arranged by motion coherence and prior width", "Zero means the report exactly matched the true motion direction."),
        guide_block("behavior_prior_figure"),
        svg_block("behavior_prior_figure", "figure_01b_behavior_relative_prior.svg", "Figure 2b. Responses aligned to the prior", "Twelve histograms of response minus prior center, arranged by motion coherence and prior width", "Zero means the report lay at the learned 225-degree center."),
        {"id": "model_roadmap", "type": "markdown", "body": "## Model roadmap: what each comparison asks\n\nAll fitted models use the same circular responses and condition indexing. They differ in how they generate sensory-centered, prior-centered, and lapse responses, and whether trial history affects the latent choice. Independent Switching is the key baseline because it has the same three emission sources as the HMM but no temporal persistence. Serial baselines ask whether a single previous stimulus or response shifts the next report. Static and covariate HMMs allow latent states to persist."},
        {"id": "model_catalog_block", "type": "table", "tableId": "model_catalog_table"},
        {"id": "bayesian_model", "type": "markdown", "body": "### Basic Bayesian observer: integrate evidence into one posterior\n\nA noisy internal measurement supplies a likelihood over directions; the learned environmental distribution supplies a prior. Their product, after normalization, is a posterior. A mode, circular mean, or random posterior sample becomes the intended report, then motor noise and lapse probability broaden the response. This is a contextual model from the original framing and was not refitted here."},
        {"id": "bayesian_math", "type": "html", "sourceId": "implementation", "body": math_panels["bayesian_math"]},
        {"id": "original_switching_model", "type": "markdown", "body": "### Original condition-dependent Switching observer: choose one source\n\nInstead of averaging prior and likelihood into one posterior response, this observer chooses a sensory-like or prior-like response source. Choice probability changes with their relative precision. That mechanism explains how two response peaks can coexist. It is explained for scientific context but was not refitted and is not ranked in this run."},
        {"id": "original_switching_math", "type": "html", "sourceId": "implementation", "body": math_panels["original_switching_math"]},
        {"id": "independent_model", "type": "markdown", "body": "### Independent Switching: same emissions, no memory\n\nEvery trial independently receives one of three components: sensory, prior, or lapse. The component weights do not depend on the preceding trial. This makes it the cleanest control for the HMM: any held-out HMM improvement cannot be attributed merely to adding sensory/prior/lapse response shapes."},
        {"id": "independent_math", "type": "html", "sourceId": "implementation", "body": math_panels["independent_math"]},
        {"id": "serial_model", "type": "markdown", "body": "### Serial baselines: one-back attraction without latent persistence\n\nThese models shift the sensory response center toward the preceding stimulus, the preceding response, or both. Separate previous-stimulus and previous-response coefficients quantify attraction; zero means no shift. Arrays reset at run boundaries. They test a specific one-back explanation, not arbitrary long history."},
        {"id": "serial_math", "type": "html", "sourceId": "implementation", "body": math_panels["serial_math"]},
        {"id": "static_hmm_model", "type": "markdown", "body": "### Static HMM: a persistent hidden strategy sequence\n\nThe state on the current trial depends on the state on the immediately preceding trial. A 3 by 3 transition matrix governs switching among sensory, prior, and lapse states; the emission model maps each state to a circular response density. Forward-backward inference sums over every possible hidden path, and Baum-Welch EM alternates posterior state inference with parameter updates."},
        {"id": "static_hmm_math", "type": "html", "sourceId": "implementation", "body": math_panels["static_hmm_math"]},
        {"id": "covariate_hmm_model", "type": "markdown", "body": "### Covariate HMM: persistence can change with conditions\n\nThe static transition matrix is replaced by a row-wise softmax regression. Current coherence, prior precision, conflict, previous error, previous conflict, previous coherence, and within-run trial position can alter transition probabilities. Covariates are standardized using the training fold only, coefficients are L2-regularized, and one destination category is fixed as the identifiability reference."},
        {"id": "covariate_hmm_math", "type": "html", "sourceId": "implementation", "body": math_panels["covariate_hmm_math"]},
        {"id": "subject_model", "type": "markdown", "body": "### Subject-level models: heterogeneity, not a full hierarchy\n\nA separate static HMM is fit to each participant. Group means and dispersions summarize those estimates, and self-transition is compared with that subject chain's stationary-state probability. This is an empirical-Bayes style summary of separate fits, not a joint hierarchical posterior with shrinkage and propagated group uncertainty."},
        {"id": "subject_math", "type": "html", "sourceId": "implementation", "body": math_panels["subject_math"]},
        {"id": "evaluation_section", "type": "markdown", "body": "## How model quality and uncertainty are defined\n\n**Primary criterion: held-out log likelihood per trial.** Models are trained on three folds of complete run sequences and asked to assign probability density to responses in the fourth. Higher, meaning less negative, is better. Exponentiating the per-trial log-likelihood difference gives the geometric-average factor by which one model increases density at each observed response.\n\n**Paired uncertainty.** The same 388 test sequences are scored by every model. Sequence bootstrap resamples those units 1,000 times and recomputes the paired difference. Its tail proportion is the fraction of resamples at or below zero, not a conventional independently derived hypothesis-test p-value.\n\n**Secondary criteria.** AIC and BIC penalize in-sample likelihood by the number of free parameters. Restart diagnostics expose local optima and convergence. Posterior predictive checks assess absolute adequacy by simulating complete datasets separately from the final static and covariate HMMs and comparing observed summaries with model-specific simulation intervals."},
        {"id": "metric_dictionary_block", "type": "table", "tableId": "metric_dictionary_table"},
        {"id": "key_result_math", "type": "html", "sourceId": "implementation", "body": key_result_math_panel(teaching)},
        {"id": "evaluation_math", "type": "html", "sourceId": "implementation", "body": math_panels["evaluation_math"]},
        {"id": "model_results", "type": "markdown", "body": f"## Held-out model comparison: temporal structure improves prediction\n\nThe {teaching['best_model']} achieved {_fmt(teaching['best_ll'], 4)} held-out LL/trial. The static HMM improved by {_fmt(teaching['static_delta'], 4)} LL/trial (95% sequence-bootstrap interval {_fmt(teaching['static_ci_low'], 4)} to {_fmt(teaching['static_ci_high'], 4)}), equivalent to {teaching['static_density_ratio']:.3f} times the predictive density per response. The covariate HMM improved by {_fmt(teaching['covariate_delta'], 4)} (interval {_fmt(teaching['covariate_ci_low'], 4)} to {_fmt(teaching['covariate_ci_high'], 4)}), or {teaching['covariate_density_ratio']:.3f} times the density. Both intervals exclude zero in the positive direction."},
        guide_block("cv_chart"),
        {"id": "cv_block", "type": "chart", "chartId": "cv_chart"},
        guide_block("delta_chart"),
        {"id": "delta_block", "type": "chart", "chartId": "delta_chart"},
        {"id": "bootstrap_table_block", "type": "table", "tableId": "bootstrap_table"},
        {"id": "hmm_section", "type": "markdown", "body": f"## Latent-state persistence and emission meaning\n\nThe sensory self-transition probability is {teaching['A_SS']:.3f}: conditional on a sensory state, the next trial remains sensory with about {100 * teaching['A_SS']:.1f}% probability. Under a homogeneous geometric-run interpretation this corresponds to {teaching['run_S']:.1f} trials. The prior self-transition probability is {teaching['A_PP']:.3f} and similarly implies {teaching['run_P']:.1f} trials. These translations explain the transition parameters; posterior MAP run lengths are separate empirical diagnostics.\n\nState labels are also checked through emissions. Sensory concentration rises with motion coherence, while prior concentration declines as prior width grows. Larger concentration means a narrower circular density, not a larger state probability."},
        guide_block("transition_chart"),
        {"id": "transition_block", "type": "chart", "chartId": "transition_chart"},
        guide_block("emission_chart"),
        {"id": "emission_block", "type": "chart", "chartId": "emission_chart"},
        {"id": "emission_interpretation_block", "type": "table", "tableId": "emission_interpretation_table"},
        {"id": "occupancy_section", "type": "markdown", "body": "## Posterior occupancy and one representative sequence\n\nPosterior occupancy is the average posterior state probability within a condition, not a hard count of known strategies. The representative run is selected deterministically from sequence-level results, then displays posterior state probabilities across its real trial order. This makes temporal organization visible without selecting an extreme example."},
        guide_block("occupancy_figure"),
        svg_block("occupancy_figure", "figure_04_posterior_state_occupancy.svg", "Figure 5. State occupancy and representative sequence", "Condition heatmaps of posterior state occupancy and trial-by-trial posterior probabilities for a deterministic representative sequence", "The same fitted static HMM supplies both the aggregate occupancy panels and trial-level posterior probabilities."),
        {"id": "subject_section", "type": "markdown", "body": f"## Subject heterogeneity: persistence is not only a pooled estimate\n\nAcross separate participant fits, mean sensory and prior self-transition probabilities were {teaching['subject_sensory_mean']:.3f} and {teaching['subject_prior_mean']:.3f}. Their variation documents heterogeneity. Stationary probabilities answer a different question: how often a state would be occupied in the chain's long-run equilibrium. Self-transition above that baseline indicates adjacency beyond what marginal occupancy alone predicts."},
        guide_block("subject_chart"),
        {"id": "subject_block", "type": "chart", "chartId": "subject_chart"},
        {"id": "serial_section", "type": "markdown", "body": "## Serial-dependence controls: one-back shifts do not explain the HMM gain\n\nPrevious-stimulus, previous-response, and combined serial baselines changed held-out LL/trial by values near zero relative to independent Switching, and every 95% sequence-bootstrap interval included zero. This supports the narrower conclusion that the HMM advantage is not reproduced by these one-back attraction terms. It does not exclude every longer or nonlinear history effect."},
        guide_block("serial_figure"),
        svg_block("serial_figure", "figure_06_serial_dependence_controls.svg", "Figure 7. Serial-dependence controls", "Paired predictive differences for one-back serial models and fitted previous-stimulus and previous-response attraction coefficients", "Intervals quantify paired sequence-bootstrap uncertainty; alpha coefficients describe fitted one-back shifts."),
        {"id": "covariate_section", "type": "markdown", "body": f"## Covariate-dependent switching: conditional associations\n\nEach plotted contrast evaluates the fitted transition probability at plus one and minus one training standard deviation while holding other standardized covariates at zero, then subtracts low from high. Positive stay contrasts increase persistence; negative contrasts decrease it. Fold-level effects and conditional sequence-bootstrap intervals test whether direction is stable across partitions. {covariate_text}\n\nBecause latent responsibilities, emissions, and covariates are estimated rather than experimentally randomized, these are conditional associations. They are not evidence that changing error or conflict would causally change strategy."},
        guide_block("covariate_chart"),
        {"id": "covariate_block", "type": "chart", "chartId": "covariate_chart"},
        {"id": "ppc_section", "type": "markdown", "body": f"## Posterior predictive adequacy: static versus covariate HMM\n\nThe final static and covariate HMM checkpoints were each used to generate {ppc_draw_text}. Motion directions, coherence, prior means and widths, sequence boundaries, and trial order were held fixed. Latent states and responses were simulated. For the covariate HMM, each next transition used the fitted all-data scaler and recomputed previous error from the **simulated** preceding response, so no observed response entered the recursive simulation.\n\nFor each replicated dataset the pipeline recomputed five metrics in all 12 condition cells, complete response histograms, and state run lengths. The 2.5th to 97.5th simulation percentiles form PPC intervals. {ppc_model_text}. {covariate_ppc_comparison} Mean-run calibration was {run_coverage_text}.\n\n**How to interpret this check.** Superior held-out likelihood asks which model predicts real unseen responses better; PPC coverage asks whether data generated by one fitted model resemble selected summaries of the observed data. Improved coverage is useful evidence of better absolute calibration, but percentile coverage is a diagnostic rather than a formal hypothesis-test pass/fail rule. A model may win the relative comparison and still omit important structure."},
        guide_block("ppc_chart"),
        {"id": "ppc_block", "type": "chart", "chartId": "ppc_chart"},
        {"id": "ppc_coverage_table_block", "type": "table", "tableId": "ppc_coverage_table"},
        guide_block("ppc_static_stimulus_figure"),
        svg_block("ppc_static_stimulus_figure", "figure_09a_ppc_relative_stimulus.svg", "Figure 10a. Static-HMM response shapes relative to stimulus", "Observed and static-HMM simulated response histograms relative to stimulus across twelve conditions", f"Bands show the simulation interval across {ppc_draw_text}."),
        guide_block("ppc_static_prior_figure"),
        svg_block("ppc_static_prior_figure", "figure_09b_ppc_relative_prior.svg", "Figure 10b. Static-HMM response shapes relative to prior", "Observed and static-HMM simulated response histograms relative to prior across twelve conditions", "These panels diagnose distribution shape, not just mean error."),
        guide_block("ppc_covariate_stimulus_figure"),
        svg_block("ppc_covariate_stimulus_figure", "figure_09c_ppc_relative_stimulus.svg", "Figure 10c. Covariate-HMM response shapes relative to stimulus", "Observed and covariate-HMM simulated response histograms relative to stimulus across twelve conditions", "The recursion uses simulated previous errors and the final fit's all-data transition scaler."),
        guide_block("ppc_covariate_prior_figure"),
        svg_block("ppc_covariate_prior_figure", "figure_09d_ppc_relative_prior.svg", "Figure 10d. Covariate-HMM response shapes relative to prior", "Observed and covariate-HMM simulated response histograms relative to prior across twelve conditions", "These model-specific panels use the same bins and observed design as the static-HMM panels."),
        guide_block("run_chart"),
        {"id": "run_block", "type": "chart", "chartId": "run_chart"},
        *sp_blocks,
        {"id": "robustness_section", "type": "markdown", "body": "## Convergence, restart stability, and exact fit criteria\n\nEach fold/model fit, final fit, and subject fit used deterministic multistart optimization; the selected restart is the one with the greatest training likelihood. The report gate exposes non-convergence instead of silently accepting it. The default run selected converged fits. A wide restart likelihood range indicates multiple local optima even when the selected fit converges, which is why 25 restarts are retained for publication estimates. Held-out likelihood remains primary; AIC and BIC are secondary all-data summaries."},
        {"id": "restart_table_block", "type": "table", "tableId": "restart_table"},
        {"id": "model_info_table_block", "type": "table", "tableId": "model_info_table"},
        {"id": "notation_section", "type": "markdown", "body": "## Terminology and notation glossary\n\nUse this table when opening the derivations. Sensory, prior, and lapse are inferred model states; they are not direct observations. Gamma and xi denote posterior probabilities, while the transition matrix is generative."},
        {"id": "notation_table_block", "type": "table", "tableId": "notation_table"},
        {"id": "figure_guide_section", "type": "markdown", "body": "## Complete visual index\n\nThis audit table proves that every native chart and numbered embedded figure has a question, reading guide, numerical takeaway, and caveat. The same entries appear adjacent to the corresponding visuals."},
        {"id": "figure_guide_table_block", "type": "table", "tableId": "figure_guide_table"},
        {"id": "limitations_section", "type": "markdown", "body": f"## Supported conclusions and unresolved limitations\n\n**Supported conclusion.** Static and covariate HMMs predict held-out responses better than independent Switching, with positive paired sequence-bootstrap intervals. Sensory- and prior-state persistence is high in the pooled fit and varies across separately fitted participants. The tested one-back serial baselines provide negligible improvement.\n\n**Unresolved limitation.** Posterior predictive calibration is incomplete: {ppc_model_text}. {covariate_ppc_comparison} {sp_change_text if sp_available else ''} Covariate effects are conditional rather than causal. State labels are model-based constructs, marginal-MAP state sequences are not observed strategies or Viterbi paths, and subject summaries are not a full hierarchical posterior. The complete PPC and conditional S/P-only sensitivity evaluate selected summaries and therefore cannot prove that either generative model is globally adequate.\n\n**Evidence boundary.** This is behavioral and computational evidence. It does not directly measure neural states, identify a biological switching mechanism, or justify a neural claim. The contextual Basic Bayesian and original condition-dependent Switching observers were not refitted and are absent from the numerical ranking."},
        {"id": "next_steps", "type": "markdown", "body": "## What would make the explanation stronger\n\nA richer dynamic model should address the PPC failure before the winning HMM is treated as an adequate process account. New experiments could vary prior means, manipulate environmental volatility, and randomize feedback availability or reliability. A full hierarchical HMM would propagate participant-level uncertainty. Neural measurements would be a separate empirical test of the computational state hypothesis, not a conclusion from this behavioral dataset."},
    ]
    if tables["restart_summary"].empty:
        tables_spec = [table for table in tables_spec if table["id"] != "restart_table"]
        blocks = [block for block in blocks if block.get("tableId") != "restart_table"]

    coverage_issues = validate_exposition_coverage(
        charts=charts,
        blocks=blocks,
        guide=guide,
        model_table=tables["model_catalog"],
    )
    if coverage_issues:
        raise RuntimeError("Report exposition coverage failed: " + "; ".join(coverage_issues))

    access_issues = [] if status == "ready" else [{"id": "publication_issue", "message": "Quality assurance only: the run manifest or selected fit convergence is incomplete.", "scope": "report status"}]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Temporally Persistent Strategy Arbitration in Human Perceptual Inference",
            "description": "Layered teaching and technical report for the Hidden Markov Switching Observer reanalysis.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables_spec,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": status,
            "datasets": _artifact_datasets(tables),
            "accessIssues": access_issues,
        },
        "sources": sources,
    }
    artifact_path = report_dir / "artifact.json"
    atomic_write_json(artifact, artifact_path)
    chart_map = _chart_map(guide)
    chart_map.to_csv(report_dir / "chart_map.csv", index=False)
    source_notes = [
        "# Report source notes",
        "",
        f"Generated: {generated_at}",
        f"Snapshot status: {status}",
        "",
        "## Audience and structure",
        "",
        "The visible report is written for a mixed audience. Full model derivations are native MathML in keyboard-accessible disclosure panels collapsed by default.",
        "Every graph has an adjacent question, encoding guide, uncertainty description, numerical takeaway, and caveat.",
        "Teaching dictionaries are generated from implementation definitions and final output tables under report/data/.",
        "",
        "## Omitted analyses",
        "",
        "The optional Basic Bayesian and original condition-dependent Switching models are explained for context but were intentionally excluded from the authoritative numerical comparison.",
        f"The posterior predictive check uses {ppc_draw_text} for both final HMMs. Coverage is diagnostic rather than a formal pass/fail decision; publication readiness does not imply absolute model adequacy.",
        *(
            [
                "The supplementary S/P-only sensitivity reuses the same simulations and excludes smoothed marginal-MAP lapse classifications symmetrically from observed and simulated response metrics.",
                f"{sp_change_text} The complete all-trial PPC remains primary, and the all-state run-length check is unchanged.",
            ]
            if sp_available
            else []
        ),
        "",
        "## Publication issues",
        "",
        *(f"- {issue}" for issue in issues),
    ]
    (report_dir / "source_notes.md").write_text("\n".join(source_notes), encoding="utf-8")
    return artifact_path


def find_report_builder() -> tuple[Path, Path]:
    plugin_base = Path.home() / ".codex" / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    candidates = sorted(plugin_base.glob("*/skills/build-report/scripts/deliver_portable_artifact.mjs"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("The Data Analytics portable report builder is not installed")
    script = candidates[0]
    plugin_root = script.parents[3]
    return plugin_root, script


_REPORT_BROWSER_STYLE_ID = "perceptual-report-overflow-fix"
_REPORT_BROWSER_STYLE = (
    f'<style id="{_REPORT_BROWSER_STYLE_ID}">'
    ".analytics-top-bar,.portable-page-header{"
    "width:100%!important;margin-left:0!important;margin-right:0!important}"
    "@media(max-width:600px){"
    "[data-layout-block-id=ppc_block] .chart-legend,"
    "[data-layout-block-id=sp_ppc_block] .chart-legend,"
    "[data-layout-block-id=run_block] .chart-legend{"
    "width:100%!important;max-width:100%!important;flex-wrap:wrap!important;"
    "justify-content:center!important;gap:4px 8px!important}"
    "[data-layout-block-id=ppc_block] .chart-legend-item,"
    "[data-layout-block-id=sp_ppc_block] .chart-legend-item,"
    "[data-layout-block-id=run_block] .chart-legend-item{"
    "margin:0!important;max-width:100%!important}"
    "[data-layout-block-id=ppc_block] .chart-legend-button,"
    "[data-layout-block-id=sp_ppc_block] .chart-legend-button,"
    "[data-layout-block-id=run_block] .chart-legend-button{white-space:normal!important}"
    "}"
    "</style>"
)


def _inject_report_browser_style(output_path: str | Path) -> None:
    output_path = Path(output_path)
    html = output_path.read_text(encoding="utf-8")
    if f'id="{_REPORT_BROWSER_STYLE_ID}"' in html:
        return
    head_close = html.lower().find("</head>")
    if head_close < 0:
        raise RuntimeError(f"Portable report has no closing head element: {output_path}")
    patched = html[:head_close] + _REPORT_BROWSER_STYLE + html[head_close:]
    temporary = output_path.with_name(f"{output_path.name}.browser-style.tmp")
    temporary.write_text(patched, encoding="utf-8")
    os.replace(temporary, output_path)


def _run_portable_browser_verifier(
    artifact_path: Path,
    output_path: Path,
    builder_script: Path,
    plugin_root: Path,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    verifier = builder_script.with_name("verify_portable_artifact.mjs")
    if not verifier.exists():
        raise FileNotFoundError(f"Portable report verifier is unavailable: {verifier}")
    screenshot = output_path.with_suffix(".verification-failure.png")
    screenshot.unlink(missing_ok=True)
    command = [
        "node",
        str(verifier),
        "--html",
        str(output_path),
        "--artifact",
        str(artifact_path),
        "--ready-timeout-ms",
        "20000",
        "--action-timeout-ms",
        "10000",
        "--timeout-ms",
        "60000",
        "--screenshot",
        str(screenshot),
    ]
    result = subprocess.run(command, cwd=plugin_root, text=True, capture_output=True, check=False)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": result.stdout.strip()}
    payload["returncode"] = result.returncode
    if result.stderr.strip():
        payload["stderr"] = result.stderr.strip()
    return payload, result


def package_report_html(artifact_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    artifact_path = Path(artifact_path).resolve()
    output_path = Path(output_path).resolve()
    plugin_root, script = find_report_builder()
    command = ["node", str(script), "--input", str(artifact_path), "--output", str(output_path)]
    result = subprocess.run(command, cwd=plugin_root, text=True, capture_output=True, check=False)
    browser_failure = None
    fallback_reason = None
    builder_output = result.stderr + result.stdout
    if result.returncode != 0 and "horizontal_overflow" in builder_output:
        browser_failure = (result.stderr or result.stdout).strip()
        fallback_reason = "shared portable reader top bar overflows by the Windows scrollbar width on long pages"
    elif (
        result.returncode != 0
        and '"stage":"static_charts"' in builder_output
        and '"code":"reader_timeout"' in builder_output
    ):
        browser_failure = (result.stderr or result.stdout).strip()
        fallback_reason = "shared portable reader static-chart probe exceeded its fixed 5-second timeout"

    if browser_failure is not None:
        fallback_env = os.environ.copy()
        fallback_env["CHROMIUM_EXECUTABLE_PATH"] = str(output_path.parent / "unavailable-browser.exe")
        result = subprocess.run(command, cwd=plugin_root, text=True, capture_output=True, check=False, env=fallback_env)
    receipt: dict[str, Any]
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError:
        receipt = {"stdout": result.stdout.strip()}
    receipt["returncode"] = result.returncode
    if browser_failure is not None:
        receipt["browser_validation_attempt"] = {
            "status": "failed",
            "reason": fallback_reason,
            "builder_output": browser_failure,
        }
    if result.stderr.strip():
        receipt["stderr"] = result.stderr.strip()
    if result.returncode != 0:
        atomic_write_json(receipt, output_path.parent / "delivery_receipt.json")
        raise RuntimeError(f"Portable report packaging failed: {result.stderr or result.stdout}")

    _inject_report_browser_style(output_path)
    verification, verification_result = _run_portable_browser_verifier(
        artifact_path,
        output_path,
        script,
        plugin_root,
    )
    verification["status"] = "passed" if verification_result.returncode == 0 else "failed"
    receipt["browser_validation"] = verification
    receipt["compatibility_patch"] = {
        "status": "applied",
        "reason": "contain the shared portable-reader top bar within the report viewport",
        "style_id": _REPORT_BROWSER_STYLE_ID,
    }
    receipt.setdefault("stages", {})["verification"] = verification["status"]
    receipt["sourceDialog"] = verification.get("sourceDialog", receipt.get("sourceDialog"))
    receipt["sourceInteraction"] = verification.get("sourceInteraction", receipt.get("sourceInteraction"))
    receipt["viewports"] = verification.get("viewports", receipt.get("viewports", []))
    if "browserWarning" in receipt:
        receipt["structural_fallback_warning"] = receipt.pop("browserWarning")
    atomic_write_json(receipt, output_path.parent / "delivery_receipt.json")
    if verification_result.returncode != 0:
        raise RuntimeError(
            "Portable report browser verification failed: "
            f"{verification_result.stderr or verification_result.stdout}"
        )
    for screenshot in output_path.parent.glob(f"{output_path.name}.tmp-*.verification-failure.png"):
        screenshot.unlink(missing_ok=True)
    return receipt


def render_publication_report(data: DataBundle, out_dir: str | Path, package_html: bool = True) -> dict[str, Any]:
    out_dir = Path(out_dir)
    tables = prepare_report_tables(data, out_dir)
    status, issues = publication_status(out_dir, tables)
    figures = generate_publication_figures(out_dir, tables)
    summary_path = generate_results_summary(out_dir)
    artifact_path = build_report_artifact(out_dir, tables, status, issues)
    html_path = out_dir / "report" / "perceptual_arbitration_results.html"
    receipt = package_report_html(artifact_path, html_path) if package_html else {}
    return {
        "status": status,
        "issues": issues,
        "tables": tables,
        "figures": figures,
        "summary_path": summary_path,
        "artifact_path": artifact_path,
        "html_path": html_path if package_html else None,
        "receipt": receipt,
    }
