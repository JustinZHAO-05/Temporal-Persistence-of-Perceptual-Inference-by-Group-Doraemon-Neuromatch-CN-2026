from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .circular import TWO_PI, circ_absdiff, circ_diff, rad2deg, wrap_rad
from .covariate_hmm import CovariateHMMParams, transition_logA_t, transition_posterior_statistics
from .data import DataBundle
from .hmm import HMMParams, K, STATE_NAMES


def parameter_count(model: str, n_coherence: int, n_prior: int, n_transition_covariates: int) -> int:
    """Return the number of free parameters used for AIC/BIC comparisons."""
    if model == "Independent_switching":
        return (K - 1) + n_coherence + n_prior
    if model == "HMM_static":
        return (K - 1) + K * (K - 1) + n_coherence + n_prior
    if model == "Covariate_HMM":
        return (K - 1) + K * (K - 1) * n_transition_covariates + n_coherence + n_prior
    if model == "Serial_stim_independent_switching":
        return parameter_count("Independent_switching", n_coherence, n_prior, n_transition_covariates) + 1
    if model == "Serial_resp_independent_switching":
        return parameter_count("Independent_switching", n_coherence, n_prior, n_transition_covariates) + 1
    if model == "Serial_both_independent_switching":
        return parameter_count("Independent_switching", n_coherence, n_prior, n_transition_covariates) + 2
    raise ValueError(f"Unknown model for parameter count: {model}")


def information_criteria(rows: list[dict], data: DataBundle) -> pd.DataFrame:
    out = []
    n = len(data.df)
    p_cov = len(data.transition_names)
    for row in rows:
        model = row["model"]
        k = parameter_count(model, len(data.coh_values), len(data.prior_values), p_cov)
        ll = float(row["train_ll"])
        out.append({
            **row,
            "n_trials": n,
            "n_parameters": k,
            "aic": 2 * k - 2 * ll,
            "bic": np.log(n) * k - 2 * ll,
        })
    return pd.DataFrame(out)


def bootstrap_model_differences(
    per_sequence_results: pd.DataFrame,
    baseline: str = "Independent_switching",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sequence-bootstrap paired held-out LL/trial differences vs baseline."""
    required = {"fold", "seq_id", "model", "test_ll", "n_test"}
    missing = required.difference(per_sequence_results.columns)
    if missing:
        raise ValueError(f"per_sequence_results is missing columns: {sorted(missing)}")

    pivot_ll = per_sequence_results.pivot_table(index=["fold", "seq_id"], columns="model", values="test_ll", aggfunc="sum")
    pivot_n = per_sequence_results.pivot_table(index=["fold", "seq_id"], values="n_test", aggfunc="first")
    joined = pivot_ll.join(pivot_n)
    if baseline not in joined.columns:
        raise ValueError(f"Baseline model {baseline!r} not present in per-sequence results")

    rng = np.random.default_rng(seed)
    rows = []
    models = [m for m in pivot_ll.columns if m != baseline]
    for model in models:
        sub = joined[[model, baseline, "n_test"]].dropna()
        if sub.empty:
            continue
        diff = (sub[model] - sub[baseline]).to_numpy(dtype=float)
        n = sub["n_test"].to_numpy(dtype=float)
        observed = float(diff.sum() / n.sum())
        draws = np.empty(int(n_bootstrap), dtype=float)
        for b in range(int(n_bootstrap)):
            sample_idx = rng.integers(0, len(sub), size=len(sub))
            draws[b] = diff[sample_idx].sum() / n[sample_idx].sum()
        rows.append({
            "model": model,
            "baseline": baseline,
            "n_sequences": len(sub),
            "observed_delta_ll_per_trial": observed,
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "bootstrap_mean": float(draws.mean()),
            "bootstrap_sd": float(draws.std(ddof=1)),
            "p_delta_le_0": float(np.mean(draws <= 0.0)),
        })
    return pd.DataFrame(rows).sort_values("observed_delta_ll_per_trial", ascending=False)


def _circular_std_deg(errors_rad: np.ndarray) -> float:
    if len(errors_rad) == 0:
        return float("nan")
    z = np.mean(np.exp(1j * errors_rad))
    r = float(np.clip(np.abs(z), 1e-12, 1.0))
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(r))))


def _condition_metrics(df: pd.DataFrame, y_rad: np.ndarray, theta_rad: np.ndarray, prior_rad: np.ndarray) -> dict:
    err = circ_diff(y_rad, theta_rad)
    abs_err = np.abs(err)
    prior_abs = circ_absdiff(y_rad, prior_rad)
    stim_abs = circ_absdiff(y_rad, theta_rad)
    return {
        "n": int(len(df)),
        "mean_abs_error_deg": float(np.rad2deg(abs_err).mean()),
        "median_abs_error_deg": float(np.rad2deg(np.median(abs_err))),
        "mean_cos_error": float(np.cos(err).mean()),
        "circular_std_error_deg": _circular_std_deg(err),
        "prior_like_rate": float(np.mean(prior_abs < stim_abs)),
    }


def simulate_hmm_responses(
    data: DataBundle,
    params: HMMParams,
    seq_ids: Iterable[int] | None = None,
    n_simulations: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate responses and latent states from a fitted static HMM."""
    if seq_ids is None:
        seq_ids = range(len(data.sequences))
    rng = np.random.default_rng(seed)
    rows = []
    for sim in range(int(n_simulations)):
        for sid in seq_ids:
            idx = data.sequences[int(sid)]
            if len(idx) == 0:
                continue
            z_prev = int(rng.choice(K, p=params.pi))
            for pos, row_idx in enumerate(idx):
                if pos > 0:
                    z_prev = int(rng.choice(K, p=params.A[z_prev]))
                if z_prev == 0:
                    mu = data.theta[row_idx]
                    kappa = float(params.kappa_s[data.coh_idx[row_idx]])
                    y = wrap_rad(rng.vonmises(mu, kappa))
                elif z_prev == 1:
                    mu = data.prior_mu[row_idx]
                    kappa = float(params.kappa_p[data.prior_idx[row_idx]])
                    y = wrap_rad(rng.vonmises(mu, kappa))
                else:
                    y = float(rng.uniform(0.0, TWO_PI))
                r = data.df.loc[row_idx]
                rows.append({
                    "simulation": sim,
                    "row_index": int(row_idx),
                    "seq_id": int(sid),
                    "subject_id": r["subject_id"],
                    "session_id": r["session_id"],
                    "run_id": r["run_id"],
                    "trial_index": r["trial_index"],
                    "motion_direction": r["motion_direction"],
                    "motion_coherence": r["motion_coherence"],
                    "prior_std": r["prior_std"],
                    "prior_mean": r["prior_mean"],
                    "estimate_deg": float(rad2deg(y)),
                    "state": STATE_NAMES[z_prev],
                })
    return pd.DataFrame(rows)


def _simulation_rng(seed: int, simulation: int) -> tuple[np.random.Generator, int]:
    """Return a deterministic draw-specific RNG independent of execution order."""
    draw_seed = int(np.random.SeedSequence([int(seed), int(simulation)]).generate_state(1, dtype=np.uint32)[0])
    return np.random.default_rng(draw_seed), draw_seed


def _sample_emission(
    rng: np.random.Generator,
    data: DataBundle,
    params: HMMParams | CovariateHMMParams,
    row_idx: int,
    state: int,
) -> float:
    if state == 0:
        mu = data.theta[row_idx]
        kappa = float(params.kappa_s[data.coh_idx[row_idx]])
        return float(wrap_rad(rng.vonmises(mu, kappa)))
    if state == 1:
        mu = data.prior_mu[row_idx]
        kappa = float(params.kappa_p[data.prior_idx[row_idx]])
        return float(wrap_rad(rng.vonmises(mu, kappa)))
    return float(rng.uniform(0.0, TWO_PI))


def _simulation_frame(
    data: DataBundle,
    simulation: int,
    draw_seed: int,
    model: str,
    y: np.ndarray,
    states: np.ndarray,
    transition_prev_error: np.ndarray,
    transition_probability: np.ndarray,
    transition_probability_sum: np.ndarray,
) -> pd.DataFrame:
    n = len(data.df)
    seq_ids = np.full(n, -1, dtype=int)
    starts = np.zeros(n, dtype=bool)
    for sid, idx in enumerate(data.sequences):
        seq_ids[idx] = sid
        starts[idx[0]] = True
    return pd.DataFrame({
        "model": model,
        "simulation": int(simulation),
        "simulation_seed": int(draw_seed),
        "row_index": np.arange(n, dtype=int),
        "seq_id": seq_ids,
        "subject_id": data.df["subject_id"].to_numpy(),
        "session_id": data.df["session_id"].to_numpy(),
        "run_id": data.df["run_id"].to_numpy(),
        "trial_index": data.df["trial_index"].to_numpy(),
        "is_sequence_start": starts,
        "motion_direction": data.df["motion_direction"].to_numpy(),
        "motion_coherence": data.df["motion_coherence"].to_numpy(),
        "prior_std": data.df["prior_std"].to_numpy(),
        "prior_mean": data.df["prior_mean"].to_numpy(),
        "estimate_deg": rad2deg(y),
        "state": np.asarray(STATE_NAMES, dtype=object)[states],
        "transition_prev_error_deg": np.rad2deg(transition_prev_error),
        "transition_probability_used": transition_probability,
        "transition_probability_row_sum": transition_probability_sum,
    })


def simulate_static_hmm_draw(
    data: DataBundle,
    params: HMMParams,
    *,
    simulation: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate one complete design-conditioned dataset from the static HMM."""
    rng, draw_seed = _simulation_rng(seed, simulation)
    n = len(data.df)
    y = np.empty(n, dtype=float)
    states = np.empty(n, dtype=int)
    transition_prev_error = np.full(n, np.nan, dtype=float)
    transition_probability = np.full(n, np.nan, dtype=float)
    transition_probability_sum = np.full(n, np.nan, dtype=float)
    for idx in data.sequences:
        state = int(rng.choice(K, p=params.pi))
        for position, row_idx_value in enumerate(idx):
            row_idx = int(row_idx_value)
            if position > 0:
                probabilities = np.asarray(params.A[state], dtype=float)
                next_state = int(rng.choice(K, p=probabilities))
                transition_probability[row_idx] = probabilities[next_state]
                transition_probability_sum[row_idx] = probabilities.sum()
                state = next_state
            states[row_idx] = state
            y[row_idx] = _sample_emission(rng, data, params, row_idx, state)
    return _simulation_frame(
        data,
        simulation,
        draw_seed,
        "HMM_static",
        y,
        states,
        transition_prev_error,
        transition_probability,
        transition_probability_sum,
    )


def simulate_covariate_hmm_draw(
    data: DataBundle,
    params: CovariateHMMParams,
    *,
    simulation: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate one dataset while recursively rebuilding response-history covariates.

    The transition into trial t uses the fitted all-data scaler. Its previous-error
    entry is recomputed from the simulated response on trial t-1; no observed
    response enters the simulated transition sequence.
    """
    rng, draw_seed = _simulation_rng(seed, simulation)
    n = len(data.df)
    y = np.empty(n, dtype=float)
    states = np.empty(n, dtype=int)
    transition_prev_error = np.full(n, np.nan, dtype=float)
    transition_probability = np.full(n, np.nan, dtype=float)
    transition_probability_sum = np.full(n, np.nan, dtype=float)
    prev_error_index = data.transition_names.index("prev_error")
    raw_prev_error_index = prev_error_index - 1

    for idx in data.sequences:
        state = int(rng.choice(K, p=params.pi))
        previous_row = -1
        for position, row_idx_value in enumerate(idx):
            row_idx = int(row_idx_value)
            if position > 0:
                simulated_prev_error = float(circ_absdiff(y[previous_row], data.theta[previous_row]))
                x_raw = data.X_transition_raw[row_idx].copy()
                x_raw[raw_prev_error_index] = simulated_prev_error
                x = np.empty(len(data.transition_names), dtype=float)
                x[0] = 1.0
                x[1:] = (x_raw - data.transition_means) / data.transition_sds
                logits = np.einsum("ijp,p->ij", params.B, x)
                logits -= np.max(logits, axis=1, keepdims=True)
                matrix = np.exp(logits)
                matrix /= matrix.sum(axis=1, keepdims=True)
                probabilities = matrix[state]
                next_state = int(rng.choice(K, p=probabilities))
                transition_prev_error[row_idx] = simulated_prev_error
                transition_probability[row_idx] = probabilities[next_state]
                transition_probability_sum[row_idx] = probabilities.sum()
                state = next_state
            states[row_idx] = state
            y[row_idx] = _sample_emission(rng, data, params, row_idx, state)
            previous_row = row_idx
    return _simulation_frame(
        data,
        simulation,
        draw_seed,
        "Covariate_HMM",
        y,
        states,
        transition_prev_error,
        transition_probability,
        transition_probability_sum,
    )


def posterior_predictive_summaries(data: DataBundle, sim_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return condition summary and relative-error histograms for observed/simulated responses."""
    obs = data.df.copy()
    obs["source"] = "observed"
    obs["simulation"] = -1
    sim = sim_df.copy()
    sim["source"] = "simulated"
    combined = pd.concat([obs, sim], ignore_index=True, sort=False)

    summary_rows = []
    hist_rows = []
    bins = np.arange(-180, 181, 15)
    for keys, g in combined.groupby(["source", "simulation", "motion_coherence", "prior_std"], sort=True):
        source, simulation, coherence, prior_std = keys
        y = np.deg2rad(np.mod(g["estimate_deg"].to_numpy(dtype=float), 360.0))
        theta = np.deg2rad(np.mod(g["motion_direction"].to_numpy(dtype=float), 360.0))
        prior = np.deg2rad(np.mod(g["prior_mean"].to_numpy(dtype=float), 360.0))
        metrics = _condition_metrics(g, y, theta, prior)
        summary_rows.append({
            "source": source,
            "simulation": int(simulation),
            "motion_coherence": coherence,
            "prior_std": prior_std,
            **metrics,
        })
        for reference, center in [("stimulus", theta), ("prior", prior)]:
            rel = np.rad2deg(circ_diff(y, center))
            counts, edges = np.histogram(rel, bins=bins)
            denom = max(int(counts.sum()), 1)
            for count, left, right in zip(counts, edges[:-1], edges[1:]):
                hist_rows.append({
                    "source": source,
                    "simulation": int(simulation),
                    "motion_coherence": coherence,
                    "prior_std": prior_std,
                    "reference": reference,
                    "bin_left_deg": float(left),
                    "bin_right_deg": float(right),
                    "count": int(count),
                    "proportion": float(count / denom),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(hist_rows)


def state_run_lengths_from_posterior(posterior_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in posterior_df.sort_values(["seq_id", "trial_index"]).groupby(["seq_id", "subject_id", "session_id", "run_id"], sort=False):
        seq_id, subject_id, session_id, run_id = keys
        states = g["state_map"].to_numpy()
        if len(states) == 0:
            continue
        start = 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start]:
                rows.append({
                    "source": "posterior_map",
                    "simulation": -1,
                    "seq_id": seq_id,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "state": states[start],
                    "run_length": i - start,
                })
                start = i
    return pd.DataFrame(rows)


def state_run_lengths_from_simulation(sim_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sort_cols = ["simulation", "seq_id", "trial_index"]
    group_cols = ["simulation", "seq_id", "subject_id", "session_id", "run_id"]
    for keys, g in sim_df.sort_values(sort_cols).groupby(group_cols, sort=False):
        simulation, seq_id, subject_id, session_id, run_id = keys
        states = g["state"].to_numpy()
        if len(states) == 0:
            continue
        start = 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start]:
                rows.append({
                    "source": "simulated",
                    "simulation": int(simulation),
                    "seq_id": seq_id,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "state": states[start],
                    "run_length": i - start,
                })
                start = i
    return pd.DataFrame(rows)


def summarize_run_lengths(run_lengths: pd.DataFrame) -> pd.DataFrame:
    if run_lengths.empty:
        return pd.DataFrame()
    return run_lengths.groupby(["source", "state"]).agg(
        n_runs=("run_length", "size"),
        mean_run_length=("run_length", "mean"),
        median_run_length=("run_length", "median"),
        q90_run_length=("run_length", lambda x: np.quantile(x, 0.90)),
    ).reset_index()


def covariate_coefficient_table(params: CovariateHMMParams, covariate_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, prev_state in enumerate(STATE_NAMES):
        for j, next_state in enumerate(STATE_NAMES):
            for p, cov in enumerate(covariate_names):
                rows.append({
                    "previous_state": prev_state,
                    "next_state": next_state,
                    "covariate": cov,
                    "coefficient": float(params.B[i, j, p]),
                })
    return pd.DataFrame(rows)


def covariate_effect_table(params: CovariateHMMParams, covariate_names: list[str]) -> pd.DataFrame:
    rows = []
    p = len(covariate_names)
    for cov_idx, cov in enumerate(covariate_names):
        if cov == "intercept":
            continue
        x_low = np.zeros(p)
        x_high = np.zeros(p)
        x_low[0] = 1.0
        x_high[0] = 1.0
        x_low[cov_idx] = -1.0
        x_high[cov_idx] = 1.0
        A_low = np.exp(transition_logA_t(params, x_low))
        A_high = np.exp(transition_logA_t(params, x_high))
        for i, prev_state in enumerate(STATE_NAMES):
            for j, next_state in enumerate(STATE_NAMES):
                rows.append({
                    "covariate": cov,
                    "previous_state": prev_state,
                    "next_state": next_state,
                    "prob_at_minus_1sd": float(A_low[i, j]),
                    "prob_at_plus_1sd": float(A_high[i, j]),
                    "delta_plus_minus": float(A_high[i, j] - A_low[i, j]),
                })
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def dataset_summary(data: DataBundle) -> pd.DataFrame:
    return pd.DataFrame([{
        "raw_rows": int(data.raw_n_rows),
        "usable_trials": int(len(data.df)),
        "dropped_rows": int(data.dropped_n_rows),
        "subjects": int(len(data.subject_values)),
        "sessions": int(data.df[["subject_id", "session_id"]].drop_duplicates().shape[0]),
        "sequences": int(len(data.sequences)),
        "coherence_levels": int(len(data.coh_values)),
        "prior_width_levels": int(len(data.prior_values)),
        "median_trials_per_sequence": float(data.seq_meta["n"].median()),
    }])


def behavioral_condition_summary(data: DataBundle) -> pd.DataFrame:
    rows = []
    for (coherence, prior_std), group in data.df.groupby(["motion_coherence", "prior_std"], sort=True):
        idx = group.index.to_numpy(dtype=int)
        metrics = _condition_metrics(group, data.y[idx], data.theta[idx], data.prior_mu[idx])
        error = circ_diff(data.y[idx], data.theta[idx])
        rows.append({
            "motion_coherence": float(coherence),
            "prior_std": float(prior_std),
            "mean_signed_error_deg": float(np.rad2deg(np.angle(np.mean(np.exp(1j * error))))),
            **metrics,
        })
    return pd.DataFrame(rows)


def observed_response_histograms(data: DataBundle, bin_width_deg: int = 15) -> pd.DataFrame:
    rows = []
    bins = np.arange(-180, 180 + bin_width_deg, bin_width_deg)
    for (coherence, prior_std), group in data.df.groupby(["motion_coherence", "prior_std"], sort=True):
        idx = group.index.to_numpy(dtype=int)
        for reference, center in [("stimulus", data.theta[idx]), ("prior", data.prior_mu[idx])]:
            relative = np.rad2deg(circ_diff(data.y[idx], center))
            counts, edges = np.histogram(relative, bins=bins)
            for count, left, right in zip(counts, edges[:-1], edges[1:]):
                rows.append({
                    "motion_coherence": float(coherence),
                    "prior_std": float(prior_std),
                    "reference": reference,
                    "bin_left_deg": float(left),
                    "bin_right_deg": float(right),
                    "bin_center_deg": float((left + right) / 2.0),
                    "count": int(count),
                    "proportion": float(count / max(len(idx), 1)),
                })
    return pd.DataFrame(rows)


def stationary_distribution(A: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(np.asarray(A, dtype=float).T)
    raw = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    if raw.sum() < 0:
        raw = -raw
    vector = np.maximum(raw, 0.0)
    if vector.sum() <= 0:
        vector = np.abs(raw)
    if vector.sum() <= 0:
        vector = np.ones(len(raw), dtype=float)
    return vector / vector.sum()


def posterior_state_occupancy(posterior_df: pd.DataFrame) -> pd.DataFrame:
    df = posterior_df.copy()
    conflict = np.abs(((df["motion_direction"] - df["prior_mean"] + 180.0) % 360.0) - 180.0)
    df["conflict_deg"] = conflict
    df["conflict_bin"] = pd.cut(
        conflict,
        bins=[-np.inf, 30.0, 60.0, np.inf],
        labels=["near (<30 deg)", "medium (30-60 deg)", "far (>=60 deg)"],
        right=False,
    ).astype(str)
    return df.groupby(["motion_coherence", "prior_std", "conflict_bin"], as_index=False, observed=True).agg(
        n=("row_index", "size"),
        p_sensory=("p_sensory", "mean"),
        p_prior=("p_prior", "mean"),
        p_lapse=("p_lapse", "mean"),
        mean_conflict_deg=("conflict_deg", "mean"),
    )


def representative_sequence(
    posterior_df: pd.DataFrame,
    per_sequence_cv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = per_sequence_cv.pivot_table(index="seq_id", columns="model", values="test_ll_per_trial", aggfunc="first")
    if not {"HMM_static", "Independent_switching"}.issubset(pivot.columns):
        return pd.DataFrame(), pd.DataFrame()
    pivot["delta_ll_per_trial"] = pivot["HMM_static"] - pivot["Independent_switching"]
    target = float(pivot["delta_ll_per_trial"].median())
    selected = int((pivot["delta_ll_per_trial"] - target).abs().sort_values(kind="stable").index[0])
    sequence = posterior_df[posterior_df["seq_id"] == selected].sort_values("trial_index").copy()
    metadata = pd.DataFrame([{
        "seq_id": selected,
        "selection_rule": "closest sequence-level HMM improvement to the median",
        "median_delta_ll_per_trial": target,
        "sequence_delta_ll_per_trial": float(pivot.loc[selected, "delta_ll_per_trial"]),
        "n_trials": int(len(sequence)),
    }])
    return sequence, metadata


def ppc_metric_intervals(ppc_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_abs_error_deg",
        "median_abs_error_deg",
        "mean_cos_error",
        "circular_std_error_deg",
        "prior_like_rate",
    ]
    rows = []
    key_cols = ["motion_coherence", "prior_std"]
    ppc_summary = ppc_summary.copy()
    for column in key_cols:
        ppc_summary[column] = pd.to_numeric(ppc_summary[column], errors="raise").round(12)
    observed = ppc_summary[ppc_summary["source"] == "observed"].set_index(key_cols)
    simulated = ppc_summary[ppc_summary["source"] == "simulated"]
    for keys, group in simulated.groupby(key_cols, sort=True):
        obs = observed.loc[keys, :]
        if isinstance(obs, pd.DataFrame):
            obs = obs.iloc[0]
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            rows.append({
                "motion_coherence": float(keys[0]),
                "prior_std": float(keys[1]),
                "condition": f"{float(keys[0]):g}/{float(keys[1]):g}",
                "metric": metric,
                "observed": float(obs[metric]),
                "simulated_mean": float(values.mean()),
                "simulated_ci_low": float(np.quantile(values, 0.025)),
                "simulated_ci_high": float(np.quantile(values, 0.975)),
                "observed_minus_simulated": float(obs[metric] - values.mean()),
            })
    return pd.DataFrame(rows)


def ppc_histogram_intervals(ppc_hist: pd.DataFrame) -> pd.DataFrame:
    keys = ["motion_coherence", "prior_std", "reference", "bin_left_deg", "bin_right_deg"]
    ppc_hist = ppc_hist.copy()
    for column in ["motion_coherence", "prior_std", "bin_left_deg", "bin_right_deg"]:
        ppc_hist[column] = pd.to_numeric(ppc_hist[column], errors="raise").round(12)
    observed = ppc_hist[ppc_hist["source"] == "observed"].set_index(keys)
    rows = []
    for key, group in ppc_hist[ppc_hist["source"] == "simulated"].groupby(keys, sort=True):
        obs = observed.loc[key, :]
        if isinstance(obs, pd.DataFrame):
            obs = obs.iloc[0]
        values = group["proportion"].to_numpy(dtype=float)
        rows.append({
            **dict(zip(keys, key)),
            "bin_center_deg": float((key[3] + key[4]) / 2.0),
            "observed": float(obs["proportion"]),
            "simulated_mean": float(values.mean()),
            "simulated_ci_low": float(np.quantile(values, 0.025)),
            "simulated_ci_high": float(np.quantile(values, 0.975)),
        })
    return pd.DataFrame(rows)


def run_length_calibration(run_lengths: pd.DataFrame) -> pd.DataFrame:
    observed = run_lengths[run_lengths["source"] == "posterior_map"]
    simulated = run_lengths[run_lengths["source"] == "simulated"]
    sim_by_draw = simulated.groupby(["simulation", "state"])["run_length"].agg(["mean", "median", lambda x: np.quantile(x, 0.9)]).reset_index()
    sim_by_draw.columns = ["simulation", "state", "mean", "median", "q90"]
    rows = []
    for state, obs in observed.groupby("state"):
        sims = sim_by_draw[sim_by_draw["state"] == state]
        for metric, obs_value in [
            ("mean", obs["run_length"].mean()),
            ("median", obs["run_length"].median()),
            ("q90", np.quantile(obs["run_length"], 0.9)),
        ]:
            values = sims[metric].to_numpy(dtype=float)
            rows.append({
                "state": state,
                "metric": metric,
                "observed": float(obs_value),
                "simulated_mean": float(values.mean()),
                "simulated_ci_low": float(np.quantile(values, 0.025)),
                "simulated_ci_high": float(np.quantile(values, 0.975)),
            })
    return pd.DataFrame(rows)


def conditional_covariate_effect_intervals(
    data: DataBundle,
    params: CovariateHMMParams,
    n_bootstrap: int = 1000,
    seed: int = 42,
    l2_transition: float = 1.0,
) -> pd.DataFrame:
    """Conditional sequence-cluster multiplier bootstrap for transition effects.

    State responsibilities and emission parameters are held fixed. Sequence-level
    score contributions are resampled with Gaussian multipliers, then transformed
    to the same +1 SD versus -1 SD probability contrasts used in the effect table.
    """
    statistics = transition_posterior_statistics(data, params, range(len(data.sequences)))
    n_sequences = len(statistics)
    if n_sequences < 2:
        raise ValueError("At least two sequences are required for clustered uncertainty")
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(int(n_bootstrap), n_sequences))
    draw_B = np.repeat(params.B[None, :, :, :], int(n_bootstrap), axis=0)
    p_features = params.B.shape[2]
    n_free_classes = K - 1

    for previous_state in range(K):
        hessian = np.zeros((n_free_classes * p_features, n_free_classes * p_features), dtype=float)
        sequence_scores = []
        beta = params.B[previous_state, :-1] - params.B[previous_state, -1]
        for _, X, xi in statistics:
            expected = xi[:, previous_state, :]
            weights = expected.sum(axis=1)
            logits = np.column_stack([X @ beta.T, np.zeros(len(X))])
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            score = ((expected[:, :-1] - weights[:, None] * probabilities[:, :-1]).T @ X).reshape(-1)
            sequence_scores.append(score)
            for a in range(n_free_classes):
                for b in range(n_free_classes):
                    weight = weights * probabilities[:, a] * ((1.0 if a == b else 0.0) - probabilities[:, b])
                    block = X.T @ (weight[:, None] * X)
                    a_slice = slice(a * p_features, (a + 1) * p_features)
                    b_slice = slice(b * p_features, (b + 1) * p_features)
                    hessian[a_slice, b_slice] += block
        hessian += l2_transition * np.eye(hessian.shape[0])
        inverse_hessian = np.linalg.pinv(hessian, rcond=1e-10)
        scores = np.vstack(sequence_scores)
        scores -= scores.mean(axis=0, keepdims=True)
        deltas = (multipliers @ scores) @ inverse_hessian.T
        beta_draws = beta.reshape(1, -1) + deltas
        for b in range(int(n_bootstrap)):
            reference = np.zeros((K, p_features), dtype=float)
            reference[:-1] = beta_draws[b].reshape(n_free_classes, p_features)
            reference -= reference.mean(axis=0, keepdims=True)
            draw_B[b, previous_state] = reference

    point = covariate_effect_table(params, data.transition_names)
    draw_rows = []
    for b in range(int(n_bootstrap)):
        draw_params = CovariateHMMParams(params.pi, draw_B[b], params.kappa_s, params.kappa_p)
        effects = covariate_effect_table(draw_params, data.transition_names)
        effects["bootstrap"] = b
        draw_rows.append(effects)
    draws = pd.concat(draw_rows, ignore_index=True)
    keys = ["covariate", "previous_state", "next_state"]
    summary = draws.groupby(keys)["delta_plus_minus"].agg(
        ci_low=lambda x: np.quantile(x, 0.025),
        ci_high=lambda x: np.quantile(x, 0.975),
        bootstrap_mean="mean",
        bootstrap_sd="std",
        p_effect_le_0=lambda x: np.mean(x <= 0.0),
    ).reset_index()
    summary["p_two_sided"] = 2.0 * np.minimum(summary["p_effect_le_0"], 1.0 - summary["p_effect_le_0"])
    result = point.merge(summary, on=keys, how="left")
    result["n_sequences"] = n_sequences
    result["method"] = "conditional sequence-cluster multiplier bootstrap"
    return result
