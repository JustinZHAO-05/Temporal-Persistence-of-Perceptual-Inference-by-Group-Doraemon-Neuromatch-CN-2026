from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def generate_results_summary(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    cv_summary = _read_csv(out_dir / "cv_summary.csv")
    cv_results = _read_csv(out_dir / "cv_results.csv")
    bootstrap = _read_csv(out_dir / "bootstrap_model_differences.csv")
    subject = _read_csv(out_dir / "subject_level_hmm.csv")
    info = _read_csv(out_dir / "model_info_criteria.csv")
    cov = _read_csv(out_dir / "covariate_hmm_effects.csv")
    ppc = _read_csv(out_dir / "posterior_predictive_condition_summary.csv")
    ppc_coverage = _read_csv(out_dir / "posterior_predictive_model_coverage.csv")
    ppc_intervals = _read_csv(out_dir / "posterior_predictive_model_metric_intervals.csv")
    sp_coverage = _read_csv(out_dir / "posterior_predictive_sp_model_coverage.csv")
    sp_retention = _read_csv(out_dir / "posterior_predictive_sp_model_retention_summary.csv")

    lines: list[str] = []
    lines.append("# Perceptual State Arbitration Results Summary")
    lines.append("")
    lines.append("This report is generated from the analysis tables in this output directory.")
    lines.append("The dataset is behavioral, so neural interpretations should be framed as computational hypotheses rather than direct neural evidence.")
    lines.append("")

    if not cv_summary.empty:
        lines.append("## Cross-validated model comparison")
        top = cv_summary.sort_values("mean_test_ll_per_trial", ascending=False).iloc[0]
        lines.append(f"Best mean held-out log likelihood per trial: **{top['model']}** at `{_fmt(top['mean_test_ll_per_trial'])}`.")
        hmm = cv_summary[cv_summary["model"] == "HMM_static"]
        ind = cv_summary[cv_summary["model"] == "Independent_switching"]
        if not hmm.empty and not ind.empty:
            delta = float(hmm.iloc[0]["mean_test_ll_per_trial"] - ind.iloc[0]["mean_test_ll_per_trial"])
            lines.append(f"Static HMM minus independent Switching: `{_fmt(delta)}` LL/trial.")
        if not bootstrap.empty:
            b = bootstrap[bootstrap["model"] == "HMM_static"]
            if not b.empty:
                r = b.iloc[0]
                lines.append(f"Sequence-bootstrap CI for HMM improvement over independent Switching: `{_fmt(r['ci_low'])}` to `{_fmt(r['ci_high'])}` LL/trial.")
        lines.append("")

    if not cv_results.empty:
        hmm_rows = cv_results[cv_results["model"] == "HMM_static"]
        if not hmm_rows.empty and {"A_SS", "A_PP"}.issubset(hmm_rows.columns):
            lines.append("## Static HMM persistence")
            lines.append(f"Mean fold-level `A_SS`: `{_fmt(hmm_rows['A_SS'].mean())}`.")
            lines.append(f"Mean fold-level `A_PP`: `{_fmt(hmm_rows['A_PP'].mean())}`.")
            lines.append("High values support temporally persistent sensory- and prior-reliance states if held-out model comparisons also favor the HMM.")
            lines.append("")

    if not subject.empty and {"A_SS", "A_PP"}.issubset(subject.columns):
        lines.append("## Subject-level fits")
        lines.append(f"Subjects fit: `{len(subject)}`.")
        lines.append(f"Mean subject `A_SS`: `{_fmt(subject['A_SS'].mean())}`; range `{_fmt(subject['A_SS'].min())}` to `{_fmt(subject['A_SS'].max())}`.")
        lines.append(f"Mean subject `A_PP`: `{_fmt(subject['A_PP'].mean())}`; range `{_fmt(subject['A_PP'].min())}` to `{_fmt(subject['A_PP'].max())}`.")
        lines.append("")

    if not cov.empty:
        lines.append("## Covariate-HMM effects")
        stay = cov[cov["previous_state"] == cov["next_state"]].copy()
        if not stay.empty:
            stay["abs_delta"] = stay["delta_plus_minus"].abs()
            for _, row in stay.sort_values("abs_delta", ascending=False).head(6).iterrows():
                lines.append(
                    f"- `{row['covariate']}` changes `{row['previous_state']}` stay probability by `{_fmt(row['delta_plus_minus'])}` from -1 SD to +1 SD."
                )
        lines.append("Previous-error effects should be interpreted conservatively unless they are large and stable.")
        lines.append("")

    if not ppc_coverage.empty:
        lines.append("## Posterior predictive checks")
        n_simulations = None
        if not ppc_intervals.empty and "n_simulations" in ppc_intervals.columns:
            n_simulations = int(ppc_intervals["n_simulations"].max())
        if n_simulations is not None:
            lines.append(f"Each final HMM was checked with `{n_simulations}` complete simulated datasets.")
        for model, frame in ppc_coverage.groupby("model"):
            lines.append(f"**{model}:** `{int(frame['covered'].sum())}/{int(frame['cells'].sum())}` condition-metric cells covered.")
            for row in frame.itertuples(index=False):
                lines.append(f"- `{row.metric}`: `{int(row.covered)}/{int(row.cells)}` cells.")
        lines.append("Coverage uses 2.5th-97.5th simulation percentiles. It is an absolute-calibration diagnostic, not a formal hypothesis-test pass/fail decision.")
        lines.append("Superior held-out likelihood and improved posterior-predictive coverage answer different questions.")
        lines.append("")
    elif not ppc.empty:
        lines.append("## Posterior predictive checks")
        lines.append("Only the legacy static-HMM PPC summary is available; inspect its figures before making adequacy claims.")
        lines.append("")

    if not sp_coverage.empty:
        lines.append("## Decoded sensory/prior-only PPC sensitivity")
        lines.append("This supplementary check excludes `L_lapse` trials using the same smoothed forward-backward marginal-MAP rule for observed and simulated responses. It is not Viterbi decoding or an online lapse detector; the complete all-trial PPC above remains primary.")
        all_totals = ppc_coverage.groupby("model", as_index=False).agg(cells=("cells", "sum"), covered=("covered", "sum"))
        sp_totals = sp_coverage.groupby("model", as_index=False).agg(cells=("cells", "sum"), covered=("covered", "sum"))
        all_lookup = {
            str(row.model): (int(row.covered), int(row.cells))
            for row in all_totals.itertuples(index=False)
        }
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
            lines.append(
                f"**{model}:** coverage {direction} from `{before_covered}/{before_cells}` all-trial cells "
                f"to `{after_covered}/{after_cells}` decoded S/P-only cells."
            )
        for model, frame in sp_retention.groupby("model"):
            observed_total = int(frame["observed_total_n"].sum())
            observed_retained = int(frame["observed_retained_n"].sum())
            simulated_retained = float(frame["simulated_retained_n_mean"].sum())
            lines.append(
                f"**{model} retention:** `{observed_retained}/{observed_total}` observed trials; "
                f"mean `{simulated_retained:.1f}/{observed_total}` simulated trials."
            )
        lines.append("Removing lapse-classified trials does not automatically establish model adequacy. The all-state run-length PPC is unchanged.")
        lines.append("")

    if not info.empty:
        lines.append("## In-sample information criteria")
        best_aic = info.sort_values("aic").iloc[0]
        best_bic = info.sort_values("bic").iloc[0]
        lines.append(f"Lowest AIC: `{best_aic['model']}`.")
        lines.append(f"Lowest BIC: `{best_bic['model']}`.")
        lines.append("Use held-out likelihood as the primary comparison; AIC/BIC are secondary in-sample diagnostics.")
        lines.append("")

    lines.append("## Interpretation rule")
    lines.append("A strong behavioral conclusion is supported only if the static or covariate HMM beats independent Switching and serial-dependence baselines in held-out likelihood, and sensory/prior self-transition probabilities remain high across folds and subjects.")
    lines.append("")

    path = out_dir / "results_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
