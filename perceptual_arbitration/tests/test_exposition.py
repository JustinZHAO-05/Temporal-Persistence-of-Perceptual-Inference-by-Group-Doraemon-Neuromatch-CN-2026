from __future__ import annotations

import base64
import re

import numpy as np
import pandas as pd

from perceptual_arbitration.diagnostics import parameter_count
from perceptual_arbitration.exposition import (
    CONTEXT_MODELS,
    FITTED_MODELS,
    embedded_svg_figure,
    figure_guide,
    implied_geometric_run_length,
    key_result_math_panel,
    kappa_to_circular_sd_deg,
    metric_dictionary,
    model_catalog,
    model_math_panels,
    notation_glossary,
    predictive_density_ratio,
    teaching_claims,
    validate_exposition_coverage,
)
from perceptual_arbitration.publication import sp_coverage_change_summary


def _result_tables(ppc_covered: bool = False) -> dict[str, pd.DataFrame]:
    center = 10.0 if ppc_covered else 20.0
    return {
        "cv_summary": pd.DataFrame({
            "model": ["Covariate_HMM", "HMM_static", "Independent_switching"],
            "model_label": ["Covariate HMM", "Static HMM", "Independent Switching"],
            "mean_test_ll_per_trial": [-0.81, -0.84, -0.91],
        }),
        "bootstrap": pd.DataFrame({
            "model": ["HMM_static", "Covariate_HMM"],
            "observed_delta_ll_per_trial": [0.06, 0.09],
            "ci_low": [0.05, 0.08],
            "ci_high": [0.07, 0.10],
        }),
        "transition": pd.DataFrame([
            {"previous_state": source, "next_state": target, "probability": value}
            for source, diagonal in [("S_sensory", 0.94), ("P_prior", 0.96), ("L_lapse", 0.68)]
            for target, value in [
                ("S_sensory", diagonal if source == "S_sensory" else (1 - diagonal) / 2),
                ("P_prior", diagonal if source == "P_prior" else (1 - diagonal) / 2),
                ("L_lapse", diagonal if source == "L_lapse" else (1 - diagonal) / 2),
            ]
        ]),
        "subject": pd.DataFrame({
            "state": ["Sensory", "Sensory", "Prior", "Prior"],
            "self_transition": [0.85, 0.90, 0.82, 0.88],
        }),
        "ppc_metrics": pd.DataFrame({
            "metric": ["mean_abs_error_deg", "prior_like_rate"],
            "observed": [10.0, 10.0],
            "simulated_ci_low": [center - 1.0, center - 1.0],
            "simulated_ci_high": [center + 1.0, center + 1.0],
        }),
        "covariate": pd.DataFrame({
            "covariate": ["coherence"],
            "previous_state": ["S_sensory"],
            "delta_plus_minus": [0.1],
        }),
    }


def test_parameter_count_formulas():
    assert parameter_count("Independent_switching", 3, 4, 8) == 9
    assert parameter_count("HMM_static", 3, 4, 8) == 15
    assert parameter_count("Covariate_HMM", 3, 4, 8) == 57
    assert parameter_count("Serial_stim_independent_switching", 3, 4, 8) == 10
    assert parameter_count("Serial_resp_independent_switching", 3, 4, 8) == 10
    assert parameter_count("Serial_both_independent_switching", 3, 4, 8) == 11


def test_intuitive_parameter_translations():
    assert np.isclose(predictive_density_ratio(0.0), 1.0)
    assert np.isclose(predictive_density_ratio(np.log(1.1)), 1.1)
    assert np.isclose(implied_geometric_run_length(0.9), 10.0)
    assert np.isinf(implied_geometric_run_length(1.0))
    assert kappa_to_circular_sd_deg(20.0) < kappa_to_circular_sd_deg(5.0)
    assert np.isinf(kappa_to_circular_sd_deg(0.0))


def test_glossary_and_model_status_are_complete():
    symbols = set(notation_glossary()["symbol"])
    assert {"yₜ", "θₜ", "zₜ", "κ", "I₀(κ)", "Aᵢⱼ", "Aₛₛ / Aₚₚ", "γₜ(i)", "ξₜ(i,j)", "LL", "ΔLL"} <= symbols

    info = pd.DataFrame({"model": ["HMM_static"], "n_parameters": [15]})
    catalog = model_catalog(info)
    assert set(catalog["model"]) == set(FITTED_MODELS + CONTEXT_MODELS)
    context = catalog[catalog["model"].isin(CONTEXT_MODELS)]
    fitted = catalog[catalog["model"].isin(FITTED_MODELS)]
    assert context["result_status"].eq("context only; not refitted").all()
    assert fitted["result_status"].eq("fitted").all()
    assert catalog.loc[catalog["model"] == "HMM_static", "parameters"].iloc[0] == 15
    sp_metric = metric_dictionary(100)
    sp_metric = sp_metric[sp_metric["metric"] == "Decoded S/P-only PPC"].iloc[0]
    assert "not Viterbi" in sp_metric["caveat"]
    assert "all-trial PPC" in sp_metric["caveat"]


def test_teaching_claims_branch_on_ppc_coverage():
    failed = teaching_claims(_result_tables(ppc_covered=False))
    covered = teaching_claims(_result_tables(ppc_covered=True))
    assert failed["all_ppc_metrics_zero_coverage"]
    assert not covered["all_ppc_metrics_zero_coverage"]
    assert failed["static_ci_low"] > 0
    assert failed["covariate_density_ratio"] > failed["static_density_ratio"] > 1


def test_teaching_claims_branch_by_ppc_model_and_mixed_metric_coverage():
    tables = _result_tables(ppc_covered=False)
    tables["ppc_metrics"] = pd.DataFrame({
        "model": ["HMM_static"] * 2 + ["Covariate_HMM"] * 2,
        "metric": ["mean_abs_error_deg", "prior_like_rate"] * 2,
        "observed": [10.0, 10.0, 10.0, 10.0],
        "simulated_ci_low": [19.0, 19.0, 9.0, 19.0],
        "simulated_ci_high": [21.0, 21.0, 11.0, 21.0],
        "n_simulations": [100] * 4,
    })
    claims = teaching_claims(tables)
    assert claims["ppc_model_zero_coverage"] == {
        "Covariate_HMM": False,
        "HMM_static": True,
    }
    assert claims["ppc_simulations"] == 100
    covariate = claims["ppc_coverage"][claims["ppc_coverage"]["model"] == "Covariate_HMM"]
    assert covariate["covered"].sum() == 1
    assert covariate["cells"].sum() == 2

    tables["ppc_metrics"].loc[tables["ppc_metrics"]["model"] == "Covariate_HMM", "simulated_ci_low"] = 19.0
    tables["ppc_metrics"].loc[tables["ppc_metrics"]["model"] == "Covariate_HMM", "simulated_ci_high"] = 21.0
    both_failed = teaching_claims(tables)
    assert both_failed["all_ppc_metrics_zero_coverage"]
    assert both_failed["all_covariate_ppc_metrics_zero_coverage"]


def test_sp_coverage_narrative_branches_from_computed_results():
    all_trial = pd.DataFrame({
        "model": ["HMM_static", "Covariate_HMM"],
        "cells": [60, 60],
        "covered": [2, 19],
    })
    sp = pd.DataFrame({
        "model": ["HMM_static", "Covariate_HMM"],
        "cells": [60, 60],
        "covered": [3, 19],
    })
    text = sp_coverage_change_summary(all_trial, sp)
    assert "improved from 2/60 to 3/60" in text
    assert "did not change from 19/60 to 19/60" in text

    sp.loc[sp["model"] == "HMM_static", "covered"] = 1
    assert "deteriorated from 2/60 to 1/60" in sp_coverage_change_summary(all_trial, sp)


def test_mathml_panels_are_collapsed_and_keyboard_accessible():
    panels = model_math_panels()
    assert set(panels) == {
        "circular_math", "bayesian_math", "original_switching_math",
        "independent_math", "serial_math", "static_hmm_math",
        "covariate_hmm_math", "subject_math", "evaluation_math",
    }
    for body in panels.values():
        assert "<details" in body and "<summary>" in body and "<math" in body
        assert not re.search(r"<details[^>]+\bopen\b", body)
        assert "summary:focus-visible" in body
        assert "Cambria Math" in body and "STIX Two Math" in body
        assert not re.search(r"<mi>(Delta|Theta|alpha|beta|gamma|kappa|lambda|mu|pi|sigma|theta|xi)</mi>", body)
        assert "<script" not in body.lower()
        assert "http://" not in body.lower() and "https://" not in body.lower()

    combined = "".join(panels.values())
    assert "<mi>θ</mi>" in combined
    assert "<mi>κ</mi>" in combined
    assert "<mo>∑</mo>" in combined
    assert "<mo>∫</mo>" in combined


def test_key_result_equations_are_visible_mathml_and_data_driven():
    claims = teaching_claims(_result_tables())
    body = key_result_math_panel(claims)
    assert body.count("<math") == 4
    assert "<details" not in body
    assert "0.940" in body and "0.960" in body
    assert f"{claims['static_density_ratio']:.3f}" in body
    assert "Cambria Math" in body
    assert "<script" not in body.lower()
    assert "http://" not in body.lower() and "https://" not in body.lower()


def test_embedded_svg_is_a_self_contained_data_uri(tmp_path):
    svg = tmp_path / "figure.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"><circle cx="1" cy="1" r="1"/></svg>', encoding="utf-8")
    body = embedded_svg_figure(svg, "Figure", "Accessible description", "Caption")
    match = re.search(r'data:image/svg\+xml;base64,([^\"]+)', body)
    assert match
    decoded = base64.b64decode(match.group(1))
    assert decoded == svg.read_bytes()
    assert 'alt="Accessible description"' in body
    assert "<script" not in body.lower()


def test_visual_and_model_exposition_coverage():
    tables = _result_tables()
    guide = figure_guide(tables)
    charts = [{"id": chart_id} for chart_id in [
        "cv_chart", "delta_chart", "transition_chart", "emission_chart",
        "subject_chart", "covariate_chart", "ppc_chart", "run_chart",
    ]]
    blocks = [{"id": f"guide_{chart['id']}"} for chart in charts]
    blocks.extend({"id": panel_id} for panel_id in model_math_panels())
    catalog = model_catalog(pd.DataFrame())
    assert validate_exposition_coverage(blocks, charts, guide, catalog) == []

    broken = [block for block in blocks if block["id"] != "guide_cv_chart"]
    assert "no adjacent explanation" in " ".join(validate_exposition_coverage(broken, charts, guide, catalog))


def test_sp_sensitivity_visuals_have_complete_guides():
    tables = _result_tables()
    tables["sp_ppc_coverage"] = pd.DataFrame({
        "model": ["HMM_static", "Covariate_HMM"],
        "metric": ["mean_abs_error_deg", "mean_abs_error_deg"],
        "cells": [12, 12],
        "covered": [0, 4],
        "coverage_rate": [0.0, 1.0 / 3.0],
    })
    guide = figure_guide(tables)
    expected = {
        "sp_coverage_figure",
        "sp_ppc_chart",
        "sp_static_stimulus_figure",
        "sp_static_prior_figure",
        "sp_covariate_stimulus_figure",
        "sp_covariate_prior_figure",
    }
    selected = guide[guide["visual_id"].isin(expected)]
    assert set(selected["visual_id"]) == expected
    assert selected["caveat"].str.contains("not Viterbi").all()
    assert selected["caveat"].str.contains("all-trial").all()
