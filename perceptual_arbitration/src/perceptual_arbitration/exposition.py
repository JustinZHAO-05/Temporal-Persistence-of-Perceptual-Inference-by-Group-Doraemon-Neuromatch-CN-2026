from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import i0e, i1e

from .data import DataBundle


FITTED_MODELS = [
    "Independent_switching",
    "Serial_stim_independent_switching",
    "Serial_resp_independent_switching",
    "Serial_both_independent_switching",
    "HMM_static",
    "Covariate_HMM",
    "Subject_level_HMM",
]

CONTEXT_MODELS = ["Basic_Bayesian", "Original_condition_switching"]

MODEL_LABELS = {
    "Basic_Bayesian": "Basic Bayesian observer",
    "Original_condition_switching": "Original condition-dependent Switching observer",
    "Independent_switching": "Independent Switching mixture",
    "Serial_stim_independent_switching": "Serial stimulus baseline",
    "Serial_resp_independent_switching": "Serial response baseline",
    "Serial_both_independent_switching": "Serial stimulus + response baseline",
    "HMM_static": "Static Hidden Markov Switching Observer",
    "Covariate_HMM": "Covariate-dependent HMM",
    "Subject_level_HMM": "Subject-level static HMM",
}


def predictive_density_ratio(delta_log_likelihood_per_trial: float) -> float:
    """Convert a per-trial log-likelihood difference to a density ratio."""
    return float(math.exp(float(delta_log_likelihood_per_trial)))


def implied_geometric_run_length(stay_probability: float) -> float:
    """Expected run length for a homogeneous Markov state with stay probability p."""
    p = float(stay_probability)
    if not 0.0 <= p < 1.0:
        return float("inf") if p == 1.0 else float("nan")
    return float(1.0 / (1.0 - p))


def kappa_to_circular_sd_deg(kappa: float) -> float:
    """Return circular standard deviation in degrees for a von Mises kappa."""
    kappa = float(kappa)
    if kappa <= 0.0:
        return float("inf")
    resultant = float(i1e(kappa) / i0e(kappa))
    resultant = float(np.clip(resultant, 1e-300, 1.0))
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(resultant))))


def data_dictionary(data: DataBundle) -> pd.DataFrame:
    rows = [
        ("estimate_x", "CSV", "Horizontal component of the response direction vector", "unitless", "Used with estimate_y to recover the reported angle"),
        ("estimate_y", "CSV", "Vertical component of the response direction vector", "unitless", "Used with estimate_x to recover the reported angle"),
        ("estimate_deg", "derived", "Reported motion direction", "degrees from 0 inclusive to 360 exclusive", "Two-argument arctangent of the response coordinates, wrapped to the circle"),
        ("motion_direction", "CSV", "True coherent motion direction on the trial", "degrees", "Sensory-state emission center"),
        ("motion_coherence", "CSV", "Fraction of dots moving coherently", "proportion", "0.06, 0.12, or 0.24; controls sensory reliability"),
        ("prior_mean", "CSV", "Center of the block direction distribution", "degrees", "Fixed at 225 degrees in the analyzed experiment"),
        ("prior_std", "CSV", "Width of the block direction distribution", "degrees", "10, 20, 40, or 80 degrees; smaller means a sharper prior"),
        ("subject_id", "CSV", "Participant identifier", "identifier", f"{len(data.subject_values)} participants"),
        ("session_id", "CSV", "Experimental session identifier", "identifier", "Sessions occurred on separate days in the original experiment"),
        ("run_id", "CSV", "Block/run identifier", "identifier", "Defines the temporal sequence boundary"),
        ("trial_index", "CSV", "Within-run trial order", "count", "Preserves temporal ordering for sequential models"),
        ("response_error", "derived", "Shortest signed response minus stimulus angle", "radians or degrees", "Always wrapped from minus 180 inclusive to 180 exclusive degrees"),
        ("conflict", "derived", "Shortest absolute stimulus-prior separation", "radians or degrees", "Current prior-sensory disagreement"),
        ("previous_error", "derived", "Absolute response error on the preceding trial in the same run", "radians", "Never crosses a run boundary"),
        ("previous_conflict", "derived", "Stimulus-prior conflict on the preceding trial in the same run", "radians", "Never crosses a run boundary"),
        ("previous_coherence", "derived", "Motion coherence on the preceding trial in the same run", "proportion", "Never crosses a run boundary"),
    ]
    return pd.DataFrame(rows, columns=["field", "origin", "meaning", "unit", "analysis_role"])


def notation_glossary() -> pd.DataFrame:
    rows = [
        ("t", "Trial index within one run-level sequence", "index"),
        ("T", "Number of trials in a sequence", "count"),
        ("yₜ", "Participant's reported direction on trial t", "angle"),
        ("θₜ", "True motion direction on trial t", "angle"),
        ("μₜ", "Learned prior center; 225 degrees here", "angle"),
        ("cₜ", "Motion coherence; 0.06, 0.12, or 0.24", "proportion"),
        ("σₜ", "Prior standard deviation; 10, 20, 40, or 80 degrees", "degrees"),
        ("zₜ", "Unobserved state on trial t", "S, P, or L"),
        ("S", "Sensory-reliance state", "state"),
        ("P", "Prior-reliance state", "state"),
        ("L", "Lapse/random-response state", "state"),
        ("κ", "von Mises concentration; larger means a narrower circular distribution", "precision"),
        ("I₀(κ)", "Modified Bessel function that normalizes a von Mises density", "normalizer"),
        ("πᵢ", "Probability that a sequence starts in state i", "probability"),
        ("Aᵢⱼ", "Probability of moving from state i to state j", "probability"),
        ("Aₛₛ / Aₚₚ", "Sensory/prior self-transition probability", "probability"),
        ("γₜ(i)", "Posterior probability that trial t is in state i", "probability"),
        ("ξₜ(i,j)", "Posterior probability of transition i to j between trials t and t+1", "probability"),
        ("αₛₜᵢₘ", "One-back attraction toward the previous stimulus", "dimensionless weight"),
        ("αᵣₑₛₚ", "One-back attraction toward the previous response", "dimensionless weight"),
        ("LL", "Log likelihood assigned to observed responses", "log density"),
        ("ΔLL", "Model LL minus the independent Switching LL", "log density per trial"),
        ("R", "Mean resultant length used to estimate κ", "0 to 1"),
    ]
    return pd.DataFrame(rows, columns=["symbol", "meaning", "unit_or_range"])


def metric_dictionary(n_ppc_simulations: int | None = None) -> pd.DataFrame:
    ppc_caveat = (
        f"Intervals use {n_ppc_simulations} complete simulated datasets per model"
        if n_ppc_simulations is not None
        else "Percentile coverage is a diagnostic, not a formal hypothesis test"
    )
    rows = [
        ("Held-out LL/trial", "Average log predictive density assigned to held-out responses", "Higher, or less negative, is better", "Primary predictive comparison"),
        ("Predictive-density ratio", "Exponential of the per-trial log-likelihood difference", "Above 1 favors the comparison model", "Geometric average density ratio, not percent accuracy"),
        ("AIC", "Twice the parameter count minus twice the in-sample log likelihood", "Lower is better", "In-sample fit penalized by 2 per free parameter"),
        ("BIC", "Log sample size times parameter count minus twice the in-sample log likelihood", "Lower is better", "Stronger sample-size-dependent complexity penalty"),
        ("Sequence-bootstrap CI", "2.5th and 97.5th percentiles after resampling run sequences", "Excluding zero supports a stable direction", "Preserves the temporal clustering unit"),
        ("Bootstrap nonpositive tail proportion", "Fraction of bootstrap differences at or below zero", "Smaller means fewer resamples reverse the effect", "Directional bootstrap tail proportion, not a standalone classical p-value"),
        ("Self-transition", "Diagonal entry of the fitted transition matrix", "Closer to 1 means greater state persistence", "Compare with the chain's stationary probability"),
        ("Implied run length", "Reciprocal of one minus self-transition probability", "Expected trials in a homogeneous geometric run", "A teaching translation, not the posterior MAP run statistic"),
        ("Prior-like rate", "Fraction of responses closer to the prior center than to the current stimulus", "Larger means more responses lie closer to the prior", "Descriptive classification, not a recovered latent state"),
        ("PPC interval", "2.5th to 97.5th percentile across complete simulated datasets", "Coverage indicates compatibility with simulated variability", ppc_caveat),
        ("Decoded S/P-only PPC", "PPC metrics after symmetrically excluding smoothed marginal-MAP lapse classifications", "Compare with the complete-model PPC rather than reading it alone", "Uses the complete response sequence, is not Viterbi decoding or an online lapse detector, and does not replace the all-trial PPC"),
    ]
    return pd.DataFrame(rows, columns=["metric", "definition", "direction", "caveat"])


def model_catalog(model_info: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    if not model_info.empty and {"model", "n_parameters"}.issubset(model_info.columns):
        counts = dict(zip(model_info["model"], model_info["n_parameters"]))
    descriptions = {
        "Basic_Bayesian": ("No", "Multiplies likelihood and prior, then reads out a posterior", "Grid-based likelihood optimization", "context only; not refitted"),
        "Original_condition_switching": ("No", "Chooses prior or sensory report from relative precision", "Grid-based likelihood optimization", "context only; not refitted"),
        "Independent_switching": ("No", "Three trial-independent sensory/prior/lapse components", "EM with deterministic multistart", "fitted"),
        "Serial_stim_independent_switching": ("One-back center shift", "Independent mixture plus previous-stimulus attraction", "L-BFGS-B multistart", "fitted"),
        "Serial_resp_independent_switching": ("One-back center shift", "Independent mixture plus previous-response attraction", "L-BFGS-B multistart", "fitted"),
        "Serial_both_independent_switching": ("One-back center shift", "Independent mixture plus both attraction terms", "L-BFGS-B multistart", "fitted"),
        "HMM_static": ("Markov", "Three emissions connected by one transition matrix", "Baum-Welch EM with deterministic multistart", "fitted"),
        "Covariate_HMM": ("Covariate-dependent Markov", "Transition probabilities vary with current and previous-trial variables", "EM plus regularized multinomial transition M-step", "fitted"),
        "Subject_level_HMM": ("Markov within subject", "One separately fitted static HMM per participant", "Baum-Welch EM; empirical-Bayes group summary", "fitted"),
    }
    rows = []
    for model in CONTEXT_MODELS + FITTED_MODELS:
        temporal, mechanism, fitting, status = descriptions[model]
        rows.append({
            "model": model,
            "label": MODEL_LABELS[model],
            "temporal_structure": temporal,
            "core_mechanism": mechanism,
            "fitting_method": fitting,
            "parameters": counts.get(model, "not counted in authoritative run"),
            "result_status": status,
        })
    return pd.DataFrame(rows)


def analysis_pipeline_table(n_ppc_simulations: int | None = None) -> pd.DataFrame:
    ppc_operation = (
        f"Simulate {n_ppc_simulations} complete datasets from each final HMM and compare metrics, histograms, and run lengths"
        if n_ppc_simulations is not None
        else "Simulate complete datasets from each final HMM and compare metrics, histograms, and run lengths"
    )
    rows = [
        (1, "Load and validate", "Raw direction CSV", "Drop rows missing required response/condition fields; recover circular response angle", "DataBundle"),
        (2, "Preserve sequences", "Subject, session, run, trial order", "Sort trials and define each subject-session-run as one sequence", "388 sequences"),
        (3, "Construct features", "Circular responses and conditions", "Compute circular errors, conflicts, previous-trial covariates, and condition indices", "Model arrays"),
        (4, "Split for prediction", "Run-level sequences", "Four-fold stratification by subject and prior width", "Train/test sequence sets"),
        (5, "Fit models", "Training sequences only", "Fixed-seed multistart optimization; retain maximum training likelihood", "Selected fold fits"),
        (6, "Score held-out trials", "Test sequences", "Evaluate predictive log density without refitting", "LL/trial and per-sequence LL"),
        (7, "Quantify uncertainty", "Paired sequence results", "Resample sequences 1,000 times and recompute model differences", "Bootstrap intervals"),
        (8, "Fit final models", "All usable sequences", "Estimate final parameters, posterior states, subject fits, and covariate effects", "Final tables"),
        (9, "Check adequacy", "Final static and covariate HMM checkpoints", ppc_operation, "Model-labelled PPC tables and figures"),
        (10, "Check decoded S/P sensitivity", "Existing PPC responses and final checkpoints", "Re-decode observed and simulated sequences with forward-backward smoothing, exclude marginal-MAP lapse trials symmetrically, and recompute response metrics without resimulation", "Conditional PPC, retention, and classification tables"),
        (11, "Render report", "Completed output tables", "Generate narrative, figures, provenance, and portable HTML without fitting", "Self-contained report"),
    ]
    return pd.DataFrame(rows, columns=["step", "stage", "input", "operation", "output"])


def emission_interpretation(emissions: pd.DataFrame) -> pd.DataFrame:
    result = emissions.copy()
    result["circular_sd_deg"] = result["value"].map(kappa_to_circular_sd_deg)
    result["plain_interpretation"] = result.apply(
        lambda row: (
            f"Responses are concentrated within a circular spread of about {row['circular_sd_deg']:.1f} degrees"
        ),
        axis=1,
    )
    return result


def _bootstrap_result(tables: dict[str, pd.DataFrame], model: str) -> pd.Series | None:
    frame = tables.get("bootstrap", pd.DataFrame())
    selected = frame[frame.get("model", pd.Series(dtype=str)) == model]
    return None if selected.empty else selected.iloc[0]


def teaching_claims(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    cv = tables["cv_summary"].sort_values("mean_test_ll_per_trial", ascending=False)
    best = cv.iloc[0]
    static = _bootstrap_result(tables, "HMM_static")
    covariate = _bootstrap_result(tables, "Covariate_HMM")
    transition = tables["transition"]
    subject = tables["subject"]
    metrics = tables["ppc_metrics"].copy()
    if "model" not in metrics.columns:
        metrics.insert(0, "model", "HMM_static")
    metrics["covered"] = (
        (metrics["observed"] >= metrics["simulated_ci_low"])
        & (metrics["observed"] <= metrics["simulated_ci_high"])
    )
    coverage = metrics.groupby(["model", "metric"], as_index=False).agg(
        cells=("observed", "size"), covered=("covered", "sum")
    )
    coverage["coverage_rate"] = coverage["covered"] / coverage["cells"]
    model_zero_coverage = {
        str(model): bool((group["covered"] == 0).all())
        for model, group in coverage.groupby("model")
    }
    n_simulations = None
    if "n_simulations" in metrics.columns and metrics["n_simulations"].notna().any():
        n_simulations = int(metrics["n_simulations"].dropna().max())
    sensory = subject[subject["state"] == "Sensory"]
    prior = subject[subject["state"] == "Prior"]

    def transition_value(previous: str, next_state: str) -> float:
        row = transition[(transition["previous_state"] == previous) & (transition["next_state"] == next_state)]
        return float(row.iloc[0]["probability"])

    return {
        "best_model": str(best["model_label"]),
        "best_ll": float(best["mean_test_ll_per_trial"]),
        "static_delta": float(static["observed_delta_ll_per_trial"]) if static is not None else float("nan"),
        "static_ci_low": float(static["ci_low"]) if static is not None else float("nan"),
        "static_ci_high": float(static["ci_high"]) if static is not None else float("nan"),
        "static_density_ratio": predictive_density_ratio(static["observed_delta_ll_per_trial"]) if static is not None else float("nan"),
        "covariate_delta": float(covariate["observed_delta_ll_per_trial"]) if covariate is not None else float("nan"),
        "covariate_ci_low": float(covariate["ci_low"]) if covariate is not None else float("nan"),
        "covariate_ci_high": float(covariate["ci_high"]) if covariate is not None else float("nan"),
        "covariate_density_ratio": predictive_density_ratio(covariate["observed_delta_ll_per_trial"]) if covariate is not None else float("nan"),
        "A_SS": transition_value("S_sensory", "S_sensory"),
        "A_PP": transition_value("P_prior", "P_prior"),
        "A_LL": transition_value("L_lapse", "L_lapse"),
        "run_S": implied_geometric_run_length(transition_value("S_sensory", "S_sensory")),
        "run_P": implied_geometric_run_length(transition_value("P_prior", "P_prior")),
        "subject_sensory_mean": float(sensory["self_transition"].mean()),
        "subject_prior_mean": float(prior["self_transition"].mean()),
        "ppc_coverage": coverage,
        "ppc_model_zero_coverage": model_zero_coverage,
        "all_ppc_metrics_zero_coverage": model_zero_coverage.get("HMM_static", bool((coverage["covered"] == 0).all())),
        "all_covariate_ppc_metrics_zero_coverage": model_zero_coverage.get("Covariate_HMM", False),
        "ppc_models": list(coverage["model"].drop_duplicates()),
        "ppc_simulations": n_simulations,
        "ppc_metric_count": int(coverage["metric"].nunique()),
        "ppc_cells_per_metric": int(coverage["cells"].min()) if not coverage.empty else 0,
    }


def figure_guide(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    claims = teaching_claims(tables)
    transition = tables["transition"]
    cov = tables["covariate"]
    strongest_cov = cov.iloc[cov["delta_plus_minus"].abs().argmax()] if not cov.empty else None
    n_simulations = claims["ppc_simulations"]
    ppc_uncertainty = (
        f"95% intervals across {n_simulations} complete simulations per model"
        if n_simulations is not None
        else "95% intervals across complete simulations"
    )
    ppc_takeaway = "; ".join(
        f"{MODEL_LABELS.get(str(model), str(model))}: {int(group['covered'].sum())}/{int(group['cells'].sum())} cells covered"
        for model, group in claims["ppc_coverage"].groupby("model")
    )
    rows = [
        ("task_figure", "Experiment schematic", "What happened on each trial and how are trials nested?", "Left-to-right task stages and dataset counts", "Pastel boxes are trial phases; arrows indicate order", "No uncertainty is shown", "The analysis preserves run-level temporal order", "This is a redraw from the paper's methods, not original artwork"),
        ("behavior_stimulus_figure", "Responses relative to the stimulus", "How tightly did reports cluster around the true direction?", "Columns are prior widths; rows are coherence; x is response minus stimulus", "Blue lines are observed proportions; zero marks the stimulus", "Descriptive histograms", "Higher coherence produces narrower stimulus-centered responses", "Distribution shape alone does not identify latent states"),
        ("behavior_prior_figure", "Responses relative to the prior", "How strongly did reports cluster around the learned prior?", "Columns are prior widths; rows are coherence; x is response minus prior", "Blue lines are observed proportions; zero marks the prior", "Descriptive histograms", "Narrow priors and weak sensory evidence increase prior-centered responses", "Prior-centered responses need not imply deliberate prior choice"),
        ("cv_chart", "Absolute held-out performance", "Which model best predicts unseen run sequences?", "Y lists models; x is mean held-out LL/trial", "Longer bars toward zero are better", "Fold standard errors appear in the publication figure", f"{claims['best_model']} is best at {claims['best_ll']:.4f} LL/trial", "Absolute LL values depend on response-density units; compare models on the same trials"),
        ("delta_chart", "Paired model differences", "How much does each model improve on independent Switching?", "X is the paired per-trial log-likelihood difference; zero means no advantage", "Positive marks favor the comparison model", "95% sequence-bootstrap intervals are in the exact table", f"Static HMM density ratio {claims['static_density_ratio']:.3f}; covariate HMM ratio {claims['covariate_density_ratio']:.3f}", "A density ratio is not classification accuracy"),
        ("transition_chart", "Static HMM transition matrix", "Do inferred states persist across adjacent trials?", "Rows are previous state; columns are next state", "Darker cells indicate larger transition probabilities", "Point estimates from the final all-data fit", f"Sensory self-transition {claims['A_SS']:.3f}; prior self-transition {claims['A_PP']:.3f}", "States are inferred computational labels, not directly observed mental states"),
        ("emission_chart", "Emission concentration", "Do fitted state labels behave as sensory and prior states should?", "X lists coherence/prior-width conditions; y is von Mises concentration", "Blue is sensory; orange is prior", "Point estimates from the final static HMM", "Sensory precision rises with coherence and prior precision falls with prior width", "Concentration is not a probability and should be translated to circular spread"),
        ("occupancy_figure", "Posterior occupancy and representative sequence", "When are sensory, prior, and lapse states inferred?", "Condition panels show posterior state probability; sequence panel follows trials", "State colors encode posterior probabilities", "Posterior probabilities condition on the fitted model", "Occupancy changes with coherence, prior width, and conflict and forms temporal runs", "The representative sequence is selected deterministically, not as the strongest example"),
        ("subject_chart", "Subject-level persistence", "Is persistence only a group-average artifact?", "X is subject; y is self-transition probability", "Filled points are self-transitions; open points in the publication figure are stationary baselines", "Separate 25-restart fits per subject", f"Mean sensory={claims['subject_sensory_mean']:.3f}; prior={claims['subject_prior_mean']:.3f}", "Separate fits are empirical-Bayes summaries, not a hierarchical posterior"),
        ("serial_figure", "Serial-dependence controls", "Can attraction to only the previous trial explain the HMM advantage?", "Model differences and fitted one-back attraction coefficients", "Intervals crossing zero indicate unstable predictive improvement", "Paired sequence-bootstrap intervals", "One-back baselines are essentially tied with independent Switching", "Small fitted coefficients do not rule out every possible history effect"),
        ("covariate_chart", "Covariate transition effects", "Which conditions change state persistence?", "Y lists covariate/state pairs; x is probability at +1 SD minus -1 SD", "Right increases staying; left decreases staying", "Conditional sequence-bootstrap intervals", (f"Largest contrast: {strongest_cov['covariate']} for {strongest_cov['previous_state']} = {strongest_cov['delta_plus_minus']:.3f}" if strongest_cov is not None else "Effects unavailable"), "Associations condition on fitted responsibilities and are not causal effects"),
        ("ppc_chart", "PPC condition metrics", "Do the static and covariate HMMs reproduce observed condition summaries?", "Rows separate models; columns show metrics; x is the coherence/prior-width cell", "Observed points are compared with simulated means and percentile intervals", ppc_uncertainty, ppc_takeaway, "Coverage is an absolute calibration diagnostic, not a formal hypothesis-test decision"),
        ("ppc_static_stimulus_figure", "Static-HMM distributions relative to stimulus", "Does the static HMM reproduce complete response shapes around the stimulus?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Shows where the static HMM captures or misses stimulus-centered shape", "Visual agreement in one cell does not imply condition-wide adequacy"),
        ("ppc_static_prior_figure", "Static-HMM distributions relative to prior", "Does the static HMM reproduce complete response shapes around the prior?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Shows condition-specific static-HMM prior-centered shape calibration", "Histogram agreement is complementary to the five metric coverage tests"),
        ("ppc_covariate_stimulus_figure", "Covariate-HMM distributions relative to stimulus", "Does covariate-dependent switching improve stimulus-centered response-shape calibration?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Directly checks the final covariate HMM under the observed design", "Improvement over the static HMM does not by itself establish absolute adequacy"),
        ("ppc_covariate_prior_figure", "Covariate-HMM distributions relative to prior", "Does the covariate HMM reproduce prior-centered response shapes?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Shows where dynamic transitions do or do not repair prior-centered mismatch", "Transition covariates are conditional predictors, not causal manipulations"),
        ("run_chart", "State run-length calibration", "Does each HMM reproduce its own inferred temporal persistence after simulation?", "Panels separate models; x is state; y is mean run length in trials", "Observed smoothed marginal-MAP runs are compared with simulated latent-state runs", ppc_uncertainty, "Each model is calibrated against its own observed posterior state summary", "Marginal-MAP sequences are forward-backward summaries, not Viterbi paths or observed ground truth"),
    ]
    sp_coverage = tables.get("sp_ppc_coverage", pd.DataFrame())
    if not sp_coverage.empty:
        all_totals = claims["ppc_coverage"].groupby("model", as_index=False).agg(
            cells=("cells", "sum"), covered=("covered", "sum")
        )
        sp_totals = sp_coverage.groupby("model", as_index=False).agg(
            cells=("cells", "sum"), covered=("covered", "sum")
        )
        all_lookup = {
            str(row.model): (int(row.covered), int(row.cells))
            for row in all_totals.itertuples(index=False)
        }
        changes = []
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
            changes.append(
                f"{MODEL_LABELS.get(model, model)} {direction}: "
                f"{before_covered}/{before_cells} to {after_covered}/{after_cells}"
            )
        sp_takeaway = "; ".join(changes)
        sp_caveat = (
            "The all-trial complete-model PPC remains primary; exclusion uses the complete sequence, "
            "is not Viterbi decoding, and cannot be an online lapse detector"
        )
        rows.extend([
            ("sp_coverage_figure", "All-trial versus decoded S/P-only coverage", "Does removing symmetrically decoded lapse trials improve condition-metric calibration?", "Panels separate models; x lists metrics; y is the fraction of 12 condition cells covered", "Blue bars are the complete-model PPC; orange bars are decoded S/P-only sensitivity results", ppc_uncertainty, sp_takeaway, sp_caveat),
            ("sp_ppc_chart", "Decoded S/P-only condition metric", "How do observed and simulated mean absolute errors compare after symmetric lapse exclusion?", "X is coherence/prior-width condition; y is mean absolute error", "Lines separate observed and simulated summaries for each model", ppc_uncertainty, sp_takeaway, sp_caveat),
            ("sp_static_stimulus_figure", "Static-HMM decoded S/P distributions relative to stimulus", "Does the static HMM reproduce stimulus-centered shapes among decoded S/P trials?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Shows shape calibration after applying the identical decoded-state rule to observed and simulated responses", sp_caveat),
            ("sp_static_prior_figure", "Static-HMM decoded S/P distributions relative to prior", "Does the static HMM reproduce prior-centered shapes among decoded S/P trials?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Shows prior-relative calibration conditional on decoded S/P membership", sp_caveat),
            ("sp_covariate_stimulus_figure", "Covariate-HMM decoded S/P distributions relative to stimulus", "Does the covariate HMM reproduce stimulus-centered shapes among decoded S/P trials?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Simulated decoding recursively rebuilds previous error from simulated responses", sp_caveat),
            ("sp_covariate_prior_figure", "Covariate-HMM decoded S/P distributions relative to prior", "Does the covariate HMM reproduce prior-centered shapes among decoded S/P trials?", "A 3 by 4 grid crosses coherence and prior width", "Blue is observed; orange is simulated; bands are simulation intervals", ppc_uncertainty, "Shows prior-relative conditional calibration without changing the fitted model", sp_caveat),
        ])
    return pd.DataFrame(rows, columns=[
        "visual_id", "title", "question", "axes", "marks", "uncertainty", "takeaway", "caveat"
    ])


def build_teaching_tables(data: DataBundle, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    claims = teaching_claims(tables)
    return {
        "data_dictionary": data_dictionary(data),
        "notation_glossary": notation_glossary(),
        "metric_dictionary": metric_dictionary(claims["ppc_simulations"]),
        "model_catalog": model_catalog(tables["model_info"]),
        "analysis_pipeline": analysis_pipeline_table(claims["ppc_simulations"]),
        "figure_guide": figure_guide(tables),
        "emission_interpretation": emission_interpretation(tables["emissions"]),
    }


_PANEL_CSS = """
<style>
  :root { color-scheme: light; }
  body { color:#252a31; font-family:Arial,Helvetica,sans-serif; font-size:15px; line-height:1.55; }
  details { border:1px solid #d9dee5; border-radius:6px; background:#fbfcfd; overflow:hidden; }
  summary { cursor:pointer; font-weight:700; padding:14px 16px; color:#174f78; background:#f3f7fa; }
  summary:focus-visible { outline:3px solid #d97706; outline-offset:-3px; }
  .panel { padding:4px 18px 18px; }
  .status { display:inline-block; margin:12px 0 4px; padding:3px 7px; border-radius:4px; font-size:12px; font-weight:700; }
  .fitted { color:#174f78; background:#e6f0f7; }
  .context { color:#7a4b00; background:#fff2d6; }
  .step { border-top:1px solid #e4e7eb; padding-top:10px; margin-top:12px; }
  .step:first-of-type { border-top:0; }
  .equation { overflow-x:auto; padding:10px 8px; margin:8px 0; background:white; border-left:3px solid #2f6690; }
  math { font-family:"Cambria Math","STIX Two Math","STIXGeneral","DejaVu Math TeX Gyre","Times New Roman",serif; font-size:1.18rem; line-height:1.35; white-space:nowrap; }
  math[display="block"] { display:block; min-width:max-content; margin:0 auto; text-align:center; }
  .reading { color:#4b5563; margin:5px 0; }
  .math-summary { border:1px solid #d9dee5; border-radius:6px; background:#fbfcfd; padding:4px 18px 18px; }
  .math-summary h3 { color:#174f78; margin-bottom:4px; }
  code { font-family:Consolas,monospace; background:#eef1f4; padding:1px 4px; border-radius:3px; }
  ul { margin-top:6px; }
</style>
"""


def _math_step(title: str, mathml: str, reading: str, explanation: str) -> str:
    return (
        '<div class="step">'
        f"<h4>{html.escape(title)}</h4>"
        f'<div class="equation">{mathml}</div>'
        f'<p class="reading"><strong>Read as:</strong> {html.escape(reading)}</p>'
        f"<p>{html.escape(explanation)}</p>"
        "</div>"
    )


_GREEK_IDENTIFIERS = {
    "Delta": "Δ",
    "Theta": "Θ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "theta": "θ",
    "xi": "ξ",
}

_ROMAN_IDENTIFIERS = {
    "AIC", "BIC", "Independent", "LL", "SD", "VM", "arg", "argmax",
    "between", "cos", "evidence", "exp", "log", "prior", "ratio", "rep",
    "resp", "sd", "sensory", "stim", "test", "train",
}


def _typeset_mathml(expr: str) -> str:
    """Normalize generated MathML tokens into conventional mathematical typography."""
    for ascii_name, symbol in _GREEK_IDENTIFIERS.items():
        expr = expr.replace(f"<mi>{ascii_name}</mi>", f"<mi>{symbol}</mi>")
    for identifier in _ROMAN_IDENTIFIERS:
        expr = expr.replace(
            f"<mi>{identifier}</mi>",
            f'<mi mathvariant="normal">{identifier}</mi>',
        )
    replacements = {
        "<mo>sum</mo>": "<mo>∑</mo>",
        "<mo>prod</mo>": "<mo>∏</mo>",
        "<mo>integral</mo>": "<mo>∫</mo>",
        "<mo>in</mo>": "<mo>∈</mo>",
        "<mo>bar</mo>": "<mo>¯</mo>",
        "<mo>hat</mo>": "<mo>^</mo>",
        "<mo>||</mo>": "<mo>∥</mo>",
        "<mo>~</mo>": "<mo>∼</mo>",
        "<mo>number</mo><mo>of</mo><mo>sequences</mo>": "<mtext>number of sequences</mtext>",
    }
    for source, target in replacements.items():
        expr = expr.replace(source, target)
    return expr


def _math(expr: str, aria: str) -> str:
    return (
        f'<math display="block" aria-label="{html.escape(aria, quote=True)}">'
        f"{_typeset_mathml(expr)}</math>"
    )


def _panel(title: str, plain: str, steps: list[str], status: str = "fitted") -> str:
    status_class = "context" if status == "context" else "fitted"
    status_text = "Context model - explained but not refitted" if status == "context" else "Model used in the authoritative run"
    return (
        _PANEL_CSS
        + f'<details aria-label="{html.escape(title, quote=True)}">'
        + f"<summary>Mathematical derivation: {html.escape(title)}</summary>"
        + '<div class="panel">'
        + f'<span class="status {status_class}">{status_text}</span>'
        + f"<p>{html.escape(plain)}</p>"
        + "".join(steps)
        + "</div></details>"
    )


def circular_math_panel() -> str:
    steps = [
        _math_step(
            "Shortest circular difference",
            _math('<mrow><mi>Delta</mi><mo>(</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>)</mo><mo>=</mo><mi>arg</mi><mo>(</mo><msup><mi>e</mi><mrow><mi>i</mi><mo>(</mo><mi>a</mi><mo>-</mo><mi>b</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow>', "Delta of a and b equals the argument of exp i times a minus b"),
            "the shortest signed turn from angle b to angle a",
            "This avoids treating 359 degrees and 1 degree as 358 degrees apart; their signed difference is minus 2 degrees.",
        ),
        _math_step(
            "von Mises density",
            _math('<mrow><mi>VM</mi><mo>(</mo><mi>y</mi><mo>;</mo><mi>m</mi><mo>,</mo><mi>kappa</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>exp</mi><mo>(</mo><mi>kappa</mi><mi>cos</mi><mo>(</mo><mi>y</mi><mo>-</mo><mi>m</mi><mo>)</mo><mo>)</mo></mrow><mrow><mn>2</mn><mi>pi</mi><msub><mi>I</mi><mn>0</mn></msub><mo>(</mo><mi>kappa</mi><mo>)</mo></mrow></mfrac></mrow>', "von Mises y given mean m and concentration kappa"),
            "a circular analogue of a normal density centered at m",
            "I0 normalizes the density. Kappa near zero is almost uniform; larger kappa produces a narrower peak.",
        ),
        _math_step(
            "Weighted concentration update",
            _math('<mrow><mi>R</mi><mo>=</mo><mfrac><mrow><munderover><mo>sum</mo><mi>t</mi><mi>T</mi></munderover><msub><mi>w</mi><mi>t</mi></msub><mi>cos</mi><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>-</mo><msub><mi>m</mi><mi>t</mi></msub><mo>)</mo></mrow><mrow><munderover><mo>sum</mo><mi>t</mi><mi>T</mi></munderover><msub><mi>w</mi><mi>t</mi></msub></mrow></mfrac><mo>,</mo><mi>kappa</mi><mo>=</mo><msup><mi>A</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mo>(</mo><mi>R</mi><mo>)</mo></mrow>', "R is the weighted mean cosine and kappa is the inverse resultant mapping"),
            "compute weighted circular agreement R, then convert it to concentration",
            "A(kappa)=I1(kappa)/I0(kappa). The implementation uses a stable standard approximation to its inverse.",
        ),
    ]
    return _panel("Circular measurements and von Mises emissions", "All response and stimulus calculations live on a circle, so ordinary subtraction and Gaussian densities are inappropriate.", steps)


def bayesian_math_panel() -> str:
    steps = [
        _math_step(
            "Noisy sensory evidence",
            _math('<mrow><msub><mi>e</mi><mi>t</mi></msub><mo>~</mo><mi>VM</mi><mo>(</mo><msub><mi>theta</mi><mi>t</mi></msub><mo>,</mo><msubsup><mi>kappa</mi><mi>c</mi><mi>E</mi></msubsup><mo>)</mo></mrow>', "e t is sampled from a von Mises around the true direction"),
            "the internal sensory measurement varies around the true motion direction",
            "Higher coherence receives its own fitted evidence concentration and should yield less variable measurements.",
        ),
        _math_step(
            "Bayes rule",
            _math('<mrow><mi>p</mi><mo>(</mo><mi>theta</mi><mo>|</mo><msub><mi>e</mi><mi>t</mi></msub><mo>)</mo><mo>=</mo><mfrac><mrow><mi>p</mi><mo>(</mo><msub><mi>e</mi><mi>t</mi></msub><mo>|</mo><mi>theta</mi><mo>)</mo><mi>p</mi><mo>(</mo><mi>theta</mi><mo>)</mo></mrow><mrow><mo>integral</mo><mi>p</mi><mo>(</mo><msub><mi>e</mi><mi>t</mi></msub><mo>|</mo><mi>u</mi><mo>)</mo><mi>p</mi><mo>(</mo><mi>u</mi><mo>)</mo><mi>d</mi><mi>u</mi></mrow></mfrac></mrow>', "posterior equals likelihood times prior divided by evidence"),
            "multiply the sensory likelihood by the learned prior and normalize",
            "A narrow prior pulls the posterior toward 225 degrees, especially when sensory evidence is weak.",
        ),
        _math_step(
            "Readout, motor noise, and lapse",
            _math('<mrow><msub><mi>theta</mi><mi>p</mi></msub><mo>=</mo><mi>argmax</mi><mi>p</mi><mo>(</mo><mi>theta</mi><mo>|</mo><msub><mi>e</mi><mi>t</mi></msub><mo>)</mo><mo>,</mo><mi>p</mi><mo>(</mo><mi>y</mi><mo>)</mo><mo>=</mo><mo>(</mo><mn>1</mn><mo>-</mo><mi>lambda</mi><mo>)</mo><mi>VM</mi><mo>(</mo><mi>y</mi><mo>;</mo><msub><mi>theta</mi><mi>p</mi></msub><mo>,</mo><msub><mi>kappa</mi><mi>m</mi></msub><mo>)</mo><mo>+</mo><mfrac><mi>lambda</mi><mrow><mn>2</mn><mi>pi</mi></mrow></mfrac></mrow>', "report the posterior mode with motor noise plus a uniform lapse component"),
            "choose a posterior summary, blur it by motor noise, and occasionally respond randomly",
            "The optional implementation also supports circular posterior means and posterior sampling. These models were not run in the authoritative comparison.",
        ),
    ]
    return _panel("Basic Bayesian observer", "The Basic Bayesian observer integrates prior and sensory evidence on every trial. It is included to explain the original scientific contrast, not as a result from this publication run.", steps, status="context")


def original_switching_math_panel() -> str:
    steps = [
        _math_step(
            "Precision-dependent source choice",
            _math('<mrow><msub><mi>p</mi><mi>prior</mi></msub><mo>=</mo><mfrac><msub><mi>kappa</mi><mi>prior</mi></msub><mrow><msub><mi>kappa</mi><mi>prior</mi></msub><mo>+</mo><msub><mi>kappa</mi><mi>evidence</mi></msub></mrow></mfrac><mo>,</mo><msub><mi>p</mi><mi>sensory</mi></msub><mo>=</mo><mn>1</mn><mo>-</mo><msub><mi>p</mi><mi>prior</mi></msub></mrow>', "prior choice probability is prior precision divided by total precision"),
            "choose the prior more often when it is sharper relative to sensory evidence",
            "Unlike Bayesian integration, this model selects one source rather than averaging the two sources into a posterior.",
        ),
        _math_step(
            "Response mixture",
            _math('<mrow><mi>p</mi><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo><mo>=</mo><mo>(</mo><mn>1</mn><mo>-</mo><mi>lambda</mi><mo>)</mo><mo>[</mo><msub><mi>p</mi><mi>sensory</mi></msub><msub><mi>f</mi><mi>S</mi></msub><mo>+</mo><msub><mi>p</mi><mi>prior</mi></msub><msub><mi>f</mi><mi>P</mi></msub><mo>]</mo><mo>+</mo><mfrac><mi>lambda</mi><mrow><mn>2</mn><mi>pi</mi></mrow></mfrac></mrow>', "response density is a sensory-prior mixture plus uniform lapses"),
            "combine the two possible report distributions according to the precision-based choice probability",
            "The original paper found this switching account captured bimodality better than its Basic Bayesian observer. This reanalysis does not reuse the paper's fitted estimates.",
        ),
    ]
    return _panel("Original condition-dependent Switching observer", "The original Switching observer is the conceptual parent of this reanalysis. It switches by condition but has no persistent hidden state across trials.", steps, status="context")


def independent_math_panel() -> str:
    steps = [
        _math_step(
            "Trial-independent mixture",
            _math('<mrow><mi>p</mi><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo><mo>=</mo><msub><mi>w</mi><mi>S</mi></msub><msub><mi>f</mi><mi>S</mi></msub><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo><mo>+</mo><msub><mi>w</mi><mi>P</mi></msub><msub><mi>f</mi><mi>P</mi></msub><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo><mo>+</mo><msub><mi>w</mi><mi>L</mi></msub><mfrac><mn>1</mn><mrow><mn>2</mn><mi>pi</mi></mrow></mfrac></mrow>', "response density is a weighted sensory, prior, and lapse mixture"),
            "flip the same three-sided probabilistic coin independently on every trial",
            "The emissions match the HMM, so HMM improvement isolates temporal structure rather than different response components.",
        ),
        _math_step(
            "EM responsibility and weight update",
            _math('<mrow><msub><mi>r</mi><mrow><mi>t</mi><mi>k</mi></mrow></msub><mo>=</mo><mfrac><mrow><msub><mi>w</mi><mi>k</mi></msub><msub><mi>f</mi><mi>k</mi></msub><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo></mrow><mrow><munderover><mo>sum</mo><mi>j</mi><mn>3</mn></munderover><msub><mi>w</mi><mi>j</mi></msub><msub><mi>f</mi><mi>j</mi></msub><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo></mrow></mfrac><mo>,</mo><msub><mi>w</mi><mi>k</mi></msub><mo>=</mo><mfrac><mrow><mo>sum</mo><msub><mi>r</mi><mrow><mi>t</mi><mi>k</mi></mrow></msub></mrow><mi>T</mi></mfrac></mrow>', "responsibility is normalized component density and each weight is mean responsibility"),
            "softly assign trials to components, then update their overall frequencies",
            "Condition-specific kappa values are updated from responsibility-weighted circular agreement.",
        ),
    ]
    return _panel("Independent Switching mixture", "This is the clean temporal null model: sensory, prior, and lapse components exist, but the current component does not depend on the previous trial.", steps)


def serial_math_panel() -> str:
    steps = [
        _math_step(
            "History-shifted sensory center",
            _math('<mrow><msubsup><mi>m</mi><mi>t</mi><mi>S</mi></msubsup><mo>=</mo><msub><mi>theta</mi><mi>t</mi></msub><mo>+</mo><msub><mi>alpha</mi><mi>stim</mi></msub><mi>Delta</mi><mo>(</mo><msub><mi>theta</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>,</mo><msub><mi>theta</mi><mi>t</mi></msub><mo>)</mo><mo>+</mo><msub><mi>alpha</mi><mi>resp</mi></msub><mi>Delta</mi><mo>(</mo><msub><mi>y</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>,</mo><msub><mi>theta</mi><mi>t</mi></msub><mo>)</mo></mrow>', "sensory center equals current stimulus plus weighted previous-stimulus and previous-response differences"),
            "move the sensory peak toward or away from the immediately preceding trial",
            "The previous arrays reset at each run boundary. Alphas are constrained to (-0.75, 0.75) by a tanh transform.",
        ),
        _math_step(
            "Serial likelihood",
            _math('<mrow><mi>LL</mi><mo>=</mo><munderover><mo>sum</mo><mi>t</mi><mi>T</mi></munderover><mi>log</mi><mo>[</mo><msub><mi>w</mi><mi>S</mi></msub><mi>VM</mi><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>;</mo><msubsup><mi>m</mi><mi>t</mi><mi>S</mi></msubsup><mo>,</mo><msubsup><mi>kappa</mi><msub><mi>c</mi><mi>t</mi></msub><mi>S</mi></msubsup><mo>)</mo><mo>+</mo><msub><mi>w</mi><mi>P</mi></msub><msub><mi>f</mi><mi>P</mi></msub><mo>+</mo><msub><mi>w</mi><mi>L</mi></msub><msub><mi>f</mi><mi>L</mi></msub><mo>]</mo></mrow>', "sum the log independent-mixture density after shifting the sensory center"),
            "fit history attraction while retaining independent component selection",
            "These baselines test one-back attraction, not a persistent multi-trial strategy state.",
        ),
    ]
    return _panel("Serial-dependence baselines", "Three comparison models add previous stimulus, previous response, or both to the independent mixture's sensory center.", steps)


def static_hmm_math_panel() -> str:
    steps = [
        _math_step(
            "Joint sequence model",
            _math('<mrow><mi>p</mi><mo>(</mo><msub><mi>y</mi><mrow><mn>1</mn><mo>:</mo><mi>T</mi></mrow></msub><mo>,</mo><msub><mi>z</mi><mrow><mn>1</mn><mo>:</mo><mi>T</mi></mrow></msub><mo>)</mo><mo>=</mo><msub><mi>pi</mi><msub><mi>z</mi><mn>1</mn></msub></msub><msub><mi>f</mi><msub><mi>z</mi><mn>1</mn></msub></msub><mo>(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>)</mo><munderover><mo>prod</mo><mrow><mi>t</mi><mo>=</mo><mn>2</mn></mrow><mi>T</mi></munderover><msub><mi>A</mi><mrow><msub><mi>z</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><msub><mi>z</mi><mi>t</mi></msub></mrow></msub><msub><mi>f</mi><msub><mi>z</mi><mi>t</mi></msub></msub><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo></mrow>', "joint probability equals initial state, first emission, and every transition times emission"),
            "start in a state, emit a response, then repeatedly transition and emit",
            "The three emissions are sensory-centered, prior-centered, and uniform lapse. A is shared across all conditions in the static HMM.",
        ),
        _math_step(
            "Forward recursion",
            _math('<mrow><msub><mi>alpha</mi><mi>t</mi></msub><mo>(</mo><mi>j</mi><mo>)</mo><mo>=</mo><msub><mi>f</mi><mi>j</mi></msub><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>)</mo><munderover><mo>sum</mo><mi>i</mi><mn>3</mn></munderover><msub><mi>alpha</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>(</mo><mi>i</mi><mo>)</mo><msub><mi>A</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow>', "forward mass for state j equals its emission times incoming forward mass"),
            "accumulate the probability of every possible state path ending in j",
            "The implementation scales each time step and uses stable log-emissions to avoid numerical underflow on long runs.",
        ),
        _math_step(
            "Backward recursion and posterior states",
            _math('<mrow><msub><mi>gamma</mi><mi>t</mi></msub><mo>(</mo><mi>i</mi><mo>)</mo><mo>=</mo><mfrac><mrow><msub><mi>alpha</mi><mi>t</mi></msub><mo>(</mo><mi>i</mi><mo>)</mo><msub><mi>beta</mi><mi>t</mi></msub><mo>(</mo><mi>i</mi><mo>)</mo></mrow><mrow><mo>sum</mo><msub><mi>alpha</mi><mi>t</mi></msub><msub><mi>beta</mi><mi>t</mi></msub></mrow></mfrac></mrow>', "gamma is normalized forward mass times backward mass"),
            "combine evidence from trials before and after t to infer the state at t",
            "Xi is the analogous normalized posterior for each adjacent state transition and drives the transition update.",
        ),
        _math_step(
            "Baum-Welch M-step",
            _math('<mrow><msub><mi>pi</mi><mi>i</mi></msub><mo>=</mo><mfrac><mrow><mo>sum</mo><msub><mi>gamma</mi><mn>1</mn></msub><mo>(</mo><mi>i</mi><mo>)</mo></mrow><mrow><mo>number</mo><mo>of</mo><mo>sequences</mo></mrow></mfrac><mo>,</mo><msub><mi>A</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><mrow><mo>sum</mo><msub><mi>xi</mi><mi>t</mi></msub><mo>(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo>)</mo></mrow><mrow><mo>sum</mo><msub><mi>gamma</mi><mi>t</mi></msub><mo>(</mo><mi>i</mi><mo>)</mo></mrow></mfrac></mrow>', "initial and transition probabilities are normalized expected counts"),
            "replace parameters with posterior-weighted expected frequencies",
            "Kappa values use the responsibility-weighted circular update shown in the circular panel. EM repeats until relative LL improvement falls below tolerance.",
        ),
    ]
    return _panel("Static Hidden Markov Switching Observer", "The HMM replaces independent trial-wise component selection with a latent Markov chain, so the current strategy is allowed to persist across several trials.", steps)


def covariate_hmm_math_panel() -> str:
    steps = [
        _math_step(
            "Standardized transition predictors",
            _math('<mrow><msub><mi>x</mi><mi>t</mi></msub><mo>=</mo><mo>[</mo><mn>1</mn><mo>,</mo><mi>z</mi><mo>(</mo><msub><mi>c</mi><mi>t</mi></msub><mo>)</mo><mo>,</mo><mi>z</mi><mo>(</mo><mn>1</mn><mo>/</mo><msub><mi>sigma</mi><mi>t</mi></msub><mo>)</mo><mo>,</mo><mi>z</mi><mo>(</mo><msub><mi>d</mi><mi>t</mi></msub><mo>)</mo><mo>,</mo><mi>z</mi><mo>(</mo><msub><mi>e</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>)</mo><mo>,</mo><mi>z</mi><mo>(</mo><msub><mi>d</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>)</mo><mo>,</mo><mi>z</mi><mo>(</mo><msub><mi>c</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>)</mo><mo>]</mo></mrow>', "x t contains an intercept and six standardized current or previous-trial variables"),
            "describe conditions available when transitioning into trial t",
            "Within each CV fold, means and standard deviations are estimated on training sequences and reused unchanged on test sequences.",
        ),
        _math_step(
            "Row-wise transition softmax",
            _math('<mrow><msub><mi>A</mi><mrow><mi>i</mi><mi>j</mi><mi>t</mi></mrow></msub><mo>=</mo><mfrac><mrow><mi>exp</mi><mo>(</mo><msubsup><mi>B</mi><mrow><mi>i</mi><mi>j</mi></mrow><mi>T</mi></msubsup><msub><mi>x</mi><mi>t</mi></msub><mo>)</mo></mrow><mrow><munderover><mo>sum</mo><mi>k</mi><mn>3</mn></munderover><mi>exp</mi><mo>(</mo><msubsup><mi>B</mi><mrow><mi>i</mi><mi>k</mi></mrow><mi>T</mi></msubsup><msub><mi>x</mi><mi>t</mi></msub><mo>)</mo></mrow></mfrac></mrow>', "transition probability is a softmax of covariate-dependent logits"),
            "let the probability of the next state change with experimental conditions and recent history",
            "For identifiability, one next-state coefficient vector is fixed during optimization; row centering is added back for readable output.",
        ),
        _math_step(
            "Regularized transition M-step",
            _math('<mrow><mi>Q</mi><mo>(</mo><mi>B</mi><mo>)</mo><mo>=</mo><munderover><mo>sum</mo><mi>t</mi><mrow><mi>T</mi><mo>-</mo><mn>1</mn></mrow></munderover><munderover><mo>sum</mo><mi>i</mi><mn>3</mn></munderover><munderover><mo>sum</mo><mi>j</mi><mn>3</mn></munderover><msub><mi>xi</mi><mi>t</mi></msub><mo>(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo>)</mo><mi>log</mi><msub><mi>A</mi><mrow><mi>i</mi><mi>j</mi><mi>t</mi></mrow></msub><mo>-</mo><mfrac><mi>lambda</mi><mn>2</mn></mfrac><msup><mrow><mo>||</mo><mi>B</mi><mo>||</mo></mrow><mn>2</mn></msup></mrow>', "maximize expected transition log likelihood minus an L2 penalty"),
            "fit weighted multinomial regressions using posterior transition counts",
            "Reported effects transform coefficients back to probability contrasts at +1 SD versus -1 SD, holding other standardized covariates at zero.",
        ),
    ]
    return _panel("Covariate-dependent HMM", "The covariate HMM retains temporal states but replaces the one-size-fits-all transition matrix with condition-dependent transition probabilities.", steps)


def subject_math_panel() -> str:
    steps = [
        _math_step(
            "Separate participant fits",
            _math('<mrow><msub><mi>Theta</mi><mi>s</mi></msub><mo>=</mo><mi>argmax</mi><munderover><mo>sum</mo><mrow><mi>r</mi><mo>in</mo><mi>s</mi></mrow><mi>R</mi></munderover><mi>log</mi><mi>p</mi><mo>(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>|</mo><msub><mi>Theta</mi><mi>s</mi></msub><mo>)</mo></mrow>', "each subject's parameters maximize likelihood over that subject's runs"),
            "fit one complete static HMM to each participant",
            "This allows transition and emission heterogeneity without pooling trials across people during fitting.",
        ),
        _math_step(
            "Stationary baseline",
            _math('<mrow><msup><mi>pi</mi><mo>*</mo></msup><mo>=</mo><msup><mi>pi</mi><mo>*</mo></msup><mi>A</mi><mo>,</mo><munderover><mo>sum</mo><mi>i</mi><mn>3</mn></munderover><msubsup><mi>pi</mi><mi>i</mi><mo>*</mo></msubsup><mo>=</mo><mn>1</mn></mrow>', "stationary probability is unchanged after multiplication by the transition matrix"),
            "compute how often a state would occur in the fitted chain's long-run equilibrium",
            "Persistence excess A_ii - pi_i* asks whether staying exceeds an independent draw from the same stationary composition.",
        ),
        _math_step(
            "Empirical-Bayes group summary",
            _math('<mrow><mover><mi>theta</mi><mo>bar</mo></mover><mo>=</mo><mfrac><mn>1</mn><mi>S</mi></mfrac><munderover><mo>sum</mo><mi>s</mi><mi>S</mi></munderover><msub><mover><mi>theta</mi><mo>hat</mo></mover><mi>s</mi></msub><mo>,</mo><msub><mi>SD</mi><mi>between</mi></msub><mo>=</mo><mi>sd</mi><mo>(</mo><msub><mover><mi>theta</mi><mo>hat</mo></mover><mi>s</mi></msub><mo>)</mo></mrow>', "group mean and between-subject spread summarize subject point estimates"),
            "summarize the distribution of independently estimated participant parameters",
            "This is not a full hierarchical posterior and does not propagate each subject fit's uncertainty into the group distribution.",
        ),
    ]
    return _panel("Subject-level HMM and empirical-Bayes summary", "Participant-specific fits test whether persistence is broadly present rather than created only by pooling heterogeneous observers.", steps)


def evaluation_math_panel() -> str:
    steps = [
        _math_step(
            "Held-out predictive score",
            _math('<mrow><msub><mi>LL</mi><mi>test</mi></msub><mo>/</mo><mi>N</mi><mo>=</mo><mfrac><mrow><munderover><mo>sum</mo><mi>r</mi><mi>R</mi></munderover><mi>log</mi><mi>p</mi><mo>(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>|</mo><msub><mover><mi>Theta</mi><mo>hat</mo></mover><mi>train</mi></msub><mo>)</mo></mrow><mrow><munderover><mo>sum</mo><mi>r</mi><mi>R</mi></munderover><msub><mi>N</mi><mi>r</mi></msub></mrow></mfrac></mrow>', "held-out score is total test log predictive density divided by test trials"),
            "ask how much probability density a train-fitted model assigns to unseen responses",
            "Four folds hold out complete run sequences stratified by subject and prior width. Higher values are better.",
        ),
        _math_step(
            "Paired difference and density ratio",
            _math('<mrow><mi>Delta</mi><mi>LL</mi><mo>=</mo><mfrac><mrow><mi>LL</mi><mo>(</mo><mi>M</mi><mo>)</mo><mo>-</mo><mi>LL</mi><mo>(</mo><mi>Independent</mi><mo>)</mo></mrow><mi>N</mi></mfrac><mo>,</mo><mi>ratio</mi><mo>=</mo><mi>exp</mi><mo>(</mo><mi>Delta</mi><mi>LL</mi><mo>)</mo></mrow>', "Delta LL compares the same trials and exp Delta LL is the geometric density ratio"),
            "translate an abstract log score into relative predictive density",
            "A ratio of 1.06 means about 6 percent higher geometric average predictive density, not 6 percentage points more correct.",
        ),
        _math_step(
            "Information criteria",
            _math('<mrow><mi>AIC</mi><mo>=</mo><mn>2</mn><mi>k</mi><mo>-</mo><mn>2</mn><mi>LL</mi><mo>,</mo><mi>BIC</mi><mo>=</mo><mi>log</mi><mo>(</mo><mi>n</mi><mo>)</mo><mi>k</mi><mo>-</mo><mn>2</mn><mi>LL</mi></mrow>', "AIC and BIC combine in-sample fit with parameter-count penalties"),
            "penalize flexible models that can fit training data more easily",
            "Held-out likelihood remains primary because it measures prediction directly; AIC and BIC are secondary diagnostics.",
        ),
        _math_step(
            "Sequence bootstrap and PPC",
            _math('<mrow><msup><mi>Delta</mi><mo>*</mo></msup><mo>=</mo><mfrac><mrow><mo>sum</mo><msubsup><mi>Delta</mi><mi>r</mi><mo>*</mo></msubsup></mrow><mrow><mo>sum</mo><msubsup><mi>N</mi><mi>r</mi><mo>*</mo></msubsup></mrow><mo>,</mo><msup><mi>y</mi><mi>rep</mi></msup><mo>~</mo><mi>p</mi><mo>(</mo><mi>y</mi><mo>|</mo><mover><mi>Theta</mi><mo>hat</mo></mover><mo>)</mo></mrow>', "resample run-level paired differences and separately simulate replicated datasets"),
            "quantify comparison uncertainty and test whether the fitted model can reproduce observed summaries",
            "Bootstrap intervals address comparative stability; PPCs address absolute adequacy. A model can win comparison and still fail PPCs.",
        ),
    ]
    return _panel("Training, model comparison, and model checking", "Model quality is defined primarily by prediction on unseen complete sequences, then audited with complexity penalties, restart diagnostics, and simulations.", steps)


def key_result_math_panel(claims: dict[str, Any]) -> str:
    """Build an always-visible MathML summary of quantities used in result prose."""
    static_ratio = float(claims["static_density_ratio"])
    covariate_ratio = float(claims["covariate_density_ratio"])
    a_ss = float(claims["A_SS"])
    a_pp = float(claims["A_PP"])
    run_s = float(claims["run_S"])
    run_p = float(claims["run_P"])
    steps = [
        _math_step(
            "Predictive-density ratio",
            _math(
                '<mrow><msub><mi>Delta</mi><mi>M</mi></msub><mo>=</mo><mfrac><mrow><mi>LL</mi><mo>(</mo><mi>M</mi><mo>)</mo><mo>-</mo><mi>LL</mi><mo>(</mo><mi>Independent</mi><mo>)</mo></mrow><mi>N</mi></mfrac><mo>,</mo><msub><mi>rho</mi><mi>M</mi></msub><mo>=</mo><mi>exp</mi><mo>(</mo><msub><mi>Delta</mi><mi>M</mi></msub><mo>)</mo></mrow>',
                "Delta M is the per-trial log-likelihood difference and rho M is its exponential",
            ),
            "exponentiate the per-trial log-likelihood advantage to obtain a geometric-average predictive-density ratio",
            f"The fitted static and covariate HMM ratios are {static_ratio:.3f} and {covariate_ratio:.3f}, respectively.",
        ),
        _math_step(
            "State persistence",
            _math(
                f'<mrow><msub><mi>A</mi><mrow><mi>S</mi><mi>S</mi></mrow></msub><mo>=</mo><mi>p</mi><mo>(</mo><msub><mi>z</mi><mi>t</mi></msub><mo>=</mo><mi>S</mi><mo>|</mo><msub><mi>z</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>=</mo><mi>S</mi><mo>)</mo><mo>=</mo><mn>{a_ss:.3f}</mn><mo>,</mo><msub><mi>A</mi><mrow><mi>P</mi><mi>P</mi></mrow></msub><mo>=</mo><mi>p</mi><mo>(</mo><msub><mi>z</mi><mi>t</mi></msub><mo>=</mo><mi>P</mi><mo>|</mo><msub><mi>z</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>=</mo><mi>P</mi><mo>)</mo><mo>=</mo><mn>{a_pp:.3f}</mn></mrow>',
                "A S S and A P P are the probabilities of remaining in the sensory and prior states",
            ),
            "read each diagonal transition entry as the probability of remaining in the same state on the next trial",
            "These are fitted conditional probabilities under the HMM, not directly observed strategy labels.",
        ),
        _math_step(
            "Implied geometric run length",
            _math(
                f'<mrow><mi>E</mi><mo>[</mo><msub><mi>L</mi><mi>i</mi></msub><mo>]</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>-</mo><msub><mi>A</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mrow></mfrac><mo>,</mo><mi>E</mi><mo>[</mo><msub><mi>L</mi><mi>S</mi></msub><mo>]</mo><mo>=</mo><mn>{run_s:.1f}</mn><mo>,</mo><mi>E</mi><mo>[</mo><msub><mi>L</mi><mi>P</mi></msub><mo>]</mo><mo>=</mo><mn>{run_p:.1f}</mn></mrow>',
                "expected run length in state i equals one divided by one minus its stay probability",
            ),
            "translate a homogeneous self-transition probability into an expected number of consecutive trials",
            "This teaching conversion is distinct from posterior MAP run lengths used in the predictive checks.",
        ),
        _math_step(
            "Covariate probability contrast",
            _math(
                '<mrow><msub><mi>Delta</mi><mrow><mi>p</mi><mo>,</mo><mi>j</mi></mrow></msub><mo>=</mo><mi>p</mi><mo>(</mo><msub><mi>z</mi><mi>t</mi></msub><mo>=</mo><mi>i</mi><mo>|</mo><msub><mi>z</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>=</mo><mi>i</mi><mo>,</mo><msub><mi>x</mi><mi>j</mi></msub><mo>=</mo><mo>+</mo><mn>1</mn><mo>)</mo><mo>-</mo><mi>p</mi><mo>(</mo><msub><mi>z</mi><mi>t</mi></msub><mo>=</mo><mi>i</mi><mo>|</mo><msub><mi>z</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>=</mo><mi>i</mi><mo>,</mo><msub><mi>x</mi><mi>j</mi></msub><mo>=</mo><mo>-</mo><mn>1</mn><mo>)</mo></mrow>',
                "Delta p j is the stay probability at plus one standard deviation minus the stay probability at minus one standard deviation",
            ),
            "compare high and low values of one standardized predictor while holding all other predictors at their training means",
            "The contrast is conditional on the fitted latent-state model and is not a causal effect.",
        ),
    ]
    return (
        _PANEL_CSS
        + '<section class="math-summary" aria-labelledby="key-result-math-title">'
        '<h3 id="key-result-math-title">Key mathematical quantities used in the results</h3>'
        '<p>These equations connect model scores and transition parameters to the reader-facing quantities reported below.</p>'
        + "".join(steps)
        + "</section>"
    )


def model_math_panels() -> dict[str, str]:
    return {
        "circular_math": circular_math_panel(),
        "bayesian_math": bayesian_math_panel(),
        "original_switching_math": original_switching_math_panel(),
        "independent_math": independent_math_panel(),
        "serial_math": serial_math_panel(),
        "static_hmm_math": static_hmm_math_panel(),
        "covariate_hmm_math": covariate_hmm_math_panel(),
        "subject_math": subject_math_panel(),
        "evaluation_math": evaluation_math_panel(),
    }


def embedded_svg_figure(path: str | Path, title: str, alt: str, caption: str) -> str:
    path = Path(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        "<style>body{font-family:Arial,Helvetica,sans-serif;color:#252a31;margin:0;}"
        "figure{margin:0;}h3{font-size:18px;margin:0 0 8px;}img{display:block;width:100%;height:auto;}"
        "figcaption{font-size:13px;line-height:1.45;color:#4b5563;margin-top:8px;}</style>"
        f"<figure><h3>{html.escape(title)}</h3>"
        f'<img src="data:image/svg+xml;base64,{encoded}" alt="{html.escape(alt, quote=True)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def visual_explanation(guide: pd.DataFrame, visual_id: str) -> str:
    row = guide[guide["visual_id"] == visual_id]
    if row.empty:
        raise KeyError(f"No visual guide for {visual_id}")
    item = row.iloc[0]
    return (
        f"### How to read: {item['title']}\n\n"
        f"**Question.** {item['question']}\n\n"
        f"**Axes and layout.** {item['axes']}\n\n"
        f"**Marks and colors.** {item['marks']}\n\n"
        f"**Uncertainty.** {item['uncertainty']}\n\n"
        f"**Result.** {item['takeaway']}\n\n"
        f"**Caution.** {item['caveat']}"
    )


def validate_exposition_coverage(
    blocks: list[dict[str, Any]],
    charts: list[dict[str, Any]],
    guide: pd.DataFrame,
    model_table: pd.DataFrame,
) -> list[str]:
    issues: list[str] = []
    block_ids = {block.get("id") for block in blocks}
    chart_ids = {chart.get("id") for chart in charts}
    guide_ids = set(guide["visual_id"])
    for chart_id in chart_ids:
        if chart_id not in guide_ids:
            issues.append(f"Native chart {chart_id} has no figure-guide row")
        if f"guide_{chart_id}" not in block_ids:
            issues.append(f"Native chart {chart_id} has no adjacent explanation block")
    expected_models = set(FITTED_MODELS + CONTEXT_MODELS)
    catalog_models = set(model_table["model"])
    if catalog_models != expected_models:
        issues.append(f"Model catalog mismatch: expected {sorted(expected_models)}, got {sorted(catalog_models)}")
    for panel_id in model_math_panels():
        if panel_id not in block_ids:
            issues.append(f"Missing mathematical panel block {panel_id}")
    return issues
