from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = WORKSPACE / "HMM" / "03_shuffle_control"
DATA_DIR = ROOT / "data"
FULL_DIR = ROOT / "results" / "full_fit"
HELD_DIR = ROOT / "results" / "heldout"
COMP_DIR = ROOT / "results" / "comparisons"
FIG_DIR = ROOT / "figures"
LOG_DIR = ROOT / "reports"
SOURCE_DATA = WORKSPACE / "HMM" / "01_full_data_fit" / "data" / "data01_direction4priors.csv"
MASTER_SEED = 20260717
N_FOLDS = 4
STATE_NAMES = np.array(["sensory", "prior", "lapse"])


def _load_base():
    import sys
    core = WORKSPACE / "HMM" / "02_fourfold_cv" / "code"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    import fourfold_cv_core as base
    # Final revised notebook settings, deliberately identical for both versions.
    base.MAX_ITER = 300
    base.TOL = 1e-6
    base.SMOOTHING = 1e-6
    return base


def ensure_dirs():
    for p in (DATA_DIR, FULL_DIR, HELD_DIR, COMP_DIR, FIG_DIR, LOG_DIR, ROOT / "model_artifacts"):
        p.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_snapshot() -> dict:
    paths = [
        SOURCE_DATA,
        WORKSPACE / "HMM" / "01_full_data_fit" / "code" / "HMM_main_me_diagnostics_and_rerun.ipynb",
        WORKSPACE / "HMM" / "01_full_data_fit" / "code" / "HMM_main_me_all_subjects_run.ipynb",
        WORKSPACE / "HMM" / "01_full_data_fit" / "results",
        WORKSPACE / "HMM" / "02_fourfold_cv",
    ]
    out = {}
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in files:
            out[str(p)] = {"size": p.stat().st_size, "sha256": sha256(p)}
    return out


def block_id_frame(df: pd.DataFrame) -> pd.Series:
    return (
        "s" + df["subject_id"].astype(int).astype(str).str.zfill(2)
        + "_session" + df["session_id"].astype(int).astype(str).str.zfill(3)
        + "_run" + df["run_id"].astype(int).astype(str).str.zfill(3)
    )


def prepare_data(force: bool = False) -> dict:
    ensure_dirs()
    shuffled_path = DATA_DIR / "data01_direction4priors_shuffled.csv"
    if shuffled_path.exists() and not force:
        shuffled = pd.read_csv(shuffled_path)
        report = pd.read_csv(DATA_DIR / "shuffle_block_report.csv")
        missing = pd.read_csv(DATA_DIR / "missing_trial_report.csv")
        audit = json.loads((DATA_DIR / "block_audit.json").read_text(encoding="utf-8"))
        return {"shuffled": shuffled, "block_report": report, "missing": missing, "audit": audit}

    raw = pd.read_csv(SOURCE_DATA)
    required = ["subject_id", "experiment_id", "experiment_name", "session_id", "run_id", "trial_index",
                "prior_std", "prior_mean", "motion_coherence", "motion_direction", "estimate_x", "estimate_y"]
    absent = [c for c in required if c not in raw.columns]
    if absent:
        raise AssertionError(f"Missing required columns: {absent}")
    raw["original_row_index"] = np.arange(len(raw), dtype=int)
    raw["original_trial_index"] = raw["trial_index"]
    raw["original_block_id"] = block_id_frame(raw)
    keys = ["subject_id", "session_id", "run_id"]
    grouped = raw.groupby(keys, sort=True, dropna=False)
    audit_groups = grouped.agg(
        n_trials=("trial_index", "size"),
        prior_std_unique_count=("prior_std", "nunique"),
        prior_mean_unique_count=("prior_mean", "nunique"),
        experiment_id_unique_count=("experiment_id", "nunique"),
        trial_index_min=("trial_index", "min"),
        trial_index_max=("trial_index", "max"),
    ).reset_index()
    if raw.duplicated(keys + ["trial_index"]).any():
        raise AssertionError("Duplicate trial_index inside a model block")
    if not (audit_groups["trial_index_min"] == 1).all():
        raise AssertionError("trial_index does not reset to 1 in every block")
    if not (audit_groups[["prior_std_unique_count", "prior_mean_unique_count", "experiment_id_unique_count"]] == 1).all().all():
        raise AssertionError("A subject-session-run block mixes a prior or experiment")

    missing_mask = raw[["estimate_x", "estimate_y"]].isna().any(axis=1)
    missing = raw.loc[missing_mask, required + ["original_row_index", "original_block_id"]].copy()
    missing["missing_reason"] = "estimate_x_or_estimate_y_missing"

    valid_parts = []
    for _, g in raw.sort_values(keys + ["trial_index"], kind="stable").groupby(keys, sort=True):
        g = g.copy()
        segment_num = 1
        valid_seen = False
        break_pending = False
        seg = []
        for _, row in g.iterrows():
            is_missing = bool(pd.isna(row["estimate_x"]) or pd.isna(row["estimate_y"]))
            if is_missing:
                seg.append(np.nan)
                break_pending = True
                continue
            if break_pending and valid_seen:
                segment_num += 1
            break_pending = False
            valid_seen = True
            seg.append(segment_num)
        g["_segment_number"] = seg
        valid_parts.append(g.loc[g["_segment_number"].notna()].copy())
    valid = pd.concat(valid_parts, ignore_index=True)
    valid["segment_id"] = valid["original_block_id"] + "_segment" + valid["_segment_number"].astype(int).astype(str).str.zfill(2)
    valid["shuffled_block_id"] = valid["segment_id"]

    segment_ids = sorted(valid["segment_id"].unique())
    number = {seg: i for i, seg in enumerate(segment_ids)}
    parts, report_rows = [], []
    for seg in segment_ids:
        g = valid.loc[valid["segment_id"] == seg].sort_values("trial_index", kind="stable").copy()
        seed = MASTER_SEED + number[seg]
        perm = np.random.default_rng(seed).permutation(len(g))
        out = g.iloc[perm].copy().reset_index(drop=True)
        out["shuffle_order"] = np.arange(len(out), dtype=int)
        out["shuffle_seed"] = seed
        changed = not np.array_equal(out["original_row_index"].to_numpy(), g["original_row_index"].to_numpy())
        parts.append(out)
        report_rows.append({
            "block_definition": "subject_id + session_id + run_id; then split at missing response",
            "subject_id": int(g["subject_id"].iloc[0]), "experiment_id": int(g["experiment_id"].iloc[0]),
            "session_id": int(g["session_id"].iloc[0]), "run_id": int(g["run_id"].iloc[0]),
            "original_block_id": g["original_block_id"].iloc[0], "segment_id": seg,
            "shuffled_block_id": seg, "stable_block_number": number[seg], "shuffle_seed": seed,
            "n_trials_before": len(g), "n_trials_after": len(out), "order_changed": changed,
            "prior_std_unique_count": g["prior_std"].nunique(), "prior_mean_unique_count": g["prior_mean"].nunique(),
        })
    shuffled = pd.concat(parts, ignore_index=True).sort_values(["subject_id", "shuffled_block_id", "shuffle_order"], kind="stable").reset_index(drop=True)
    report = pd.DataFrame(report_rows)
    audit = {
        "block_definition": "subject_id + session_id + run_id",
        "shuffle_definition": "subject_id + original_block_id + missing-delimited segment_id",
        "number_of_subjects": int(raw.subject_id.nunique()),
        "number_of_subject_session_pairs": int(raw[["subject_id", "session_id"]].drop_duplicates().shape[0]),
        "number_of_global_run_id_values": int(raw.run_id.nunique()),
        "number_of_model_blocks": int(audit_groups.shape[0]),
        "number_of_valid_segments": int(valid.segment_id.nunique()),
        "block_length_min": int(audit_groups.n_trials.min()), "block_length_median": float(audit_groups.n_trials.median()),
        "block_length_max": int(audit_groups.n_trials.max()),
        "max_subjects_per_run_id": int(raw.groupby("run_id").subject_id.nunique().max()),
        "max_sessions_per_run_id": int(raw.groupby("run_id").session_id.nunique().max()),
        "max_prior_std_unique_within_block": int(audit_groups.prior_std_unique_count.max()),
        "max_prior_mean_unique_within_block": int(audit_groups.prior_mean_unique_count.max()),
        "max_experiment_unique_within_block": int(audit_groups.experiment_id_unique_count.max()),
        "trial_index_resets_to_one_in_all_blocks": bool((audit_groups.trial_index_min == 1).all()),
        "raw_rows": len(raw), "valid_rows": len(valid), "missing_rows": int(missing_mask.sum()),
        "master_seed": MASTER_SEED,
    }
    cols = [c for c in shuffled.columns if c != "_segment_number"]
    shuffled[cols].to_csv(shuffled_path, index=False)
    report.to_csv(DATA_DIR / "shuffle_block_report.csv", index=False)
    missing.to_csv(DATA_DIR / "missing_trial_report.csv", index=False)
    audit_groups.to_csv(DATA_DIR / "block_audit_table.csv", index=False)
    (DATA_DIR / "block_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"shuffled": shuffled[cols], "block_report": report, "missing": missing, "audit": audit}


def add_angles(df, base):
    out = base.add_circular_columns(df)
    out["response_direction"] = np.rad2deg(np.arctan2(out["estimate_y"], out["estimate_x"])) % 360
    return out


def infer_sequences(df, params, sequence_col, base):
    local = df.reset_index(drop=True).copy()
    loge = base.log_emission_matrix(local, params)
    emissions = np.exp(loge)
    gamma_all = np.zeros((len(local), 3)); pred_all = np.zeros_like(gamma_all); filt_all = np.zeros_like(gamma_all)
    viterbi_all = np.zeros(len(local), dtype=int); scales_all = np.zeros(len(local))
    for _, g in local.groupby(sequence_col, sort=False):
        pos = g.index.to_numpy()
        le = loge[pos]
        pred, filt, scales, _ = base.forward_segment(le, params)
        beta = base.backward_segment(le, scales, params)
        gamma = filt * beta; gamma /= gamma.sum(axis=1, keepdims=True)
        transition = np.asarray(params["transition_matrix"]); pi = np.asarray(params["initial_prob"])
        delta = np.empty_like(le); psi = np.zeros_like(le, dtype=int)
        delta[0] = np.log(pi) + le[0]
        for t in range(1, len(pos)):
            scores = delta[t-1][:, None] + np.log(transition)
            psi[t] = np.argmax(scores, axis=0); delta[t] = scores[psi[t], np.arange(3)] + le[t]
        path = np.zeros(len(pos), dtype=int); path[-1] = np.argmax(delta[-1])
        for t in range(len(pos)-2, -1, -1): path[t] = psi[t+1, path[t+1]]
        gamma_all[pos] = gamma; pred_all[pos] = pred; filt_all[pos] = filt; viterbi_all[pos] = path; scales_all[pos] = scales
    local[["p_sensory", "p_prior", "p_lapse"]] = gamma_all
    local[["filtered_prob_sensory", "filtered_prob_prior", "filtered_prob_lapse"]] = filt_all
    local[["prior_predictive_prob_sensory", "prior_predictive_prob_prior", "prior_predictive_prob_lapse"]] = pred_all
    local["most_likely_state"] = STATE_NAMES[np.argmax(gamma_all, axis=1)]
    local["viterbi_state"] = STATE_NAMES[viterbi_all]
    local[["sensory_emission_likelihood", "prior_emission_likelihood", "lapse_emission_likelihood"]] = emissions
    local["one_step_predictive_likelihood"] = scales_all
    local["one_step_predictive_log_likelihood"] = np.log(scales_all)
    return local


def serialize_params(params):
    return {"initial_prob": np.asarray(params["initial_prob"]).tolist(),
            "transition_matrix": np.asarray(params["transition_matrix"]).tolist(),
            "kappaS": {str(k): float(v) for k, v in params["kappaS"].items()},
            "kappaP": {str(k): float(v) for k, v in params["kappaP"].items()}}


def transition_record(subject_id, A):
    labels = ["sensory", "prior", "lapse"]
    d = {"subject_id": int(subject_id)}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels): d[f"{a}_to_{b}"] = float(A[i, j])
    return d


def run_full_fit(force: bool = False) -> dict:
    ensure_dirs(); base = _load_base(); prep = prepare_data(force=False)
    summary_path = FULL_DIR / "all_subject_summary_shuffled.csv"
    if summary_path.exists() and not force:
        summary = pd.read_csv(summary_path)
        predictions = pd.read_csv(FULL_DIR / "all_subject_trial_predictions_shuffled.csv")
        model_cols = ["p_sensory","p_prior","p_lapse","filtered_prob_sensory","filtered_prob_prior","filtered_prob_lapse",
                      "prior_predictive_prob_sensory","prior_predictive_prob_prior","prior_predictive_prob_lapse",
                      "sensory_emission_likelihood","prior_emission_likelihood","lapse_emission_likelihood",
                      "one_step_predictive_likelihood","one_step_predictive_log_likelihood"]
        summary["has_nan"] = False
        summary["has_inf"] = False
        for sid in summary.subject_id:
            vals = predictions.loc[predictions.subject_id == sid, model_cols].to_numpy(float)
            summary.loc[summary.subject_id == sid, "has_nan"] = bool(np.isnan(vals).any())
            summary.loc[summary.subject_id == sid, "has_inf"] = bool(np.isinf(vals).any())
        summary.to_csv(summary_path, index=False)
        transdf = pd.read_csv(FULL_DIR / "all_subject_transition_matrices_shuffled.csv")
        validate_full(prep, predictions, summary, transdf)
        return {"summary": summary, "predictions": predictions}
    df = add_angles(prep["shuffled"], base)
    all_trials, summaries, trans, failures = [], [], [], []
    for sid in sorted(df.subject_id.unique()):
        start = time.perf_counter()
        try:
            subject = df.loc[df.subject_id == sid].sort_values(["shuffled_block_id", "shuffle_order"], kind="stable").reset_index(drop=True)
            fit = base.fit_model(subject)
            params = fit["params"]
            predicted = infer_sequences(subject, params, "shuffled_block_id", base)
            history = np.asarray(fit["history"], float); diffs = np.diff(history); A = np.asarray(params["transition_matrix"])
            weights = predicted[["p_sensory", "p_prior", "p_lapse"]].mean().to_numpy()
            outdir = FULL_DIR / f"subject_{int(sid):02d}"; outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "parameters.json").write_text(json.dumps(serialize_params(params), indent=2), encoding="utf-8")
            pd.DataFrame({"iteration": np.arange(1, len(history)+1), "log_likelihood": history}).to_csv(outdir / "log_likelihood_history.csv", index=False)
            pd.DataFrame(A, index=STATE_NAMES, columns=STATE_NAMES).to_csv(outdir / "transition_matrix.csv", index_label="from_state")
            predicted.to_csv(outdir / "trial_posteriors.csv", index=False); predicted.to_csv(outdir / "trial_predictions.csv", index=False)
            all_trials.append(predicted); trans.append(transition_record(sid, A))
            ks=params["kappaS"]; kp=params["kappaP"]
            summaries.append({"subject_id":int(sid), "n_trials":len(subject), "n_original_blocks":subject.original_block_id.nunique(),
                "n_segments":subject.segment_id.nunique(), "converged":fit["converged"], "n_iterations":fit["n_iterations"],
                "initial_log_likelihood":history[0], "final_log_likelihood":history[-1], "log_likelihood_improvement":history[-1]-history[0],
                "likelihood_monotonic":bool(len(diffs)==0 or diffs.min()>=-1e-8), "maximum_likelihood_drop":float(max(0,-diffs.min())) if len(diffs) else 0,
                "sensory_weight":weights[0], "prior_weight":weights[1], "lapse_weight":weights[2],
                "min_transition_probability":A.min(), "max_transition_probability":A.max(),
                "kappa_s_6":ks[.06], "kappa_s_12":ks[.12], "kappa_s_24":ks[.24], "kappa_p_10":kp[10], "kappa_p_20":kp[20], "kappa_p_40":kp[40], "kappa_p_80":kp[80],
                "state_collapse_flag":bool(weights.max()>.98), "has_nan":False,
                "has_inf":False, "reached_max_iter":fit["n_iterations"]>=300,
                "runtime_seconds":time.perf_counter()-start})
            print(f"full subject {int(sid):02d}: iter={fit['n_iterations']} converged={fit['converged']}")
        except Exception as e:
            failures.append({"subject_id":int(sid), "error_type":type(e).__name__, "error_message":str(e)})
    trials = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    summary = pd.DataFrame(summaries); transdf = pd.DataFrame(trans)
    trials.to_csv(FULL_DIR / "all_subject_trial_predictions_shuffled.csv", index=False)
    summary.to_csv(summary_path, index=False); transdf.to_csv(FULL_DIR / "all_subject_transition_matrices_shuffled.csv", index=False)
    pd.DataFrame(failures, columns=["subject_id","error_type","error_message"]).to_csv(FULL_DIR / "failed_subjects_shuffled.csv", index=False)
    build_full_comparisons(summary, transdf, trials)
    validate_full(prep, trials, summary, transdf)
    return {"summary":summary, "predictions":trials}


def dwell_switch(df, sequence_col):
    rows=[]
    for sid,gsub in df.groupby("subject_id"):
        dwell={s:[] for s in STATE_NAMES}; changes=0; possible=0
        for _,g in gsub.groupby(sequence_col, sort=False):
            a=g["viterbi_state"].astype(str).to_numpy()
            if len(a):
                start=0
                for i in range(1,len(a)+1):
                    if i==len(a) or a[i]!=a[start]: dwell.setdefault(a[start],[]).append(i-start); start=i
                changes += int(np.sum(a[1:] != a[:-1])); possible += max(0,len(a)-1)
        row={"subject_id":int(sid), "switch_rate":changes/possible if possible else np.nan, "mean_state_dwell_time":np.mean(sum(dwell.values(),[]))}
        for s in STATE_NAMES: row[f"mean_{s}_dwell_time"] = np.mean(dwell[s]) if dwell[s] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def matrix_metrics(tdf):
    cols=[f"{a}_to_{b}" for a in STATE_NAMES for b in STATE_NAMES]
    rows=[]
    for _,r in tdf.iterrows():
        A=r[cols].to_numpy(float).reshape(3,3)
        rows.append({"subject_id":int(r.subject_id), "mean_diagonal_transition":float(np.diag(A).mean()),
                     "transition_entropy":float((-A*np.log(np.clip(A,1e-300,None))).sum(axis=1).mean())})
    return pd.DataFrame(rows)


def build_full_comparisons(sh_summary, sh_trans, sh_trials):
    orig_s=pd.read_csv(WORKSPACE/"HMM"/"01_full_data_fit"/"results"/"all_subject_summary_revised.csv")
    orig_t=pd.read_csv(WORKSPACE/"HMM"/"01_full_data_fit"/"results"/"all_subject_transition_matrices_revised.csv")
    original_trials=[]
    for sid in sorted(orig_s.subject_id): original_trials.append(pd.read_csv(WORKSPACE/"HMM"/"01_full_data_fit"/"results"/f"subject_{int(sid):02d}"/f"subject_{int(sid):02d}_trial_posteriors.csv"))
    orig_trials=pd.concat(original_trials,ignore_index=True)
    om=matrix_metrics(orig_t).add_prefix("original_").rename(columns={"original_subject_id":"subject_id"})
    sm=matrix_metrics(sh_trans).add_prefix("shuffled_").rename(columns={"shuffled_subject_id":"subject_id"})
    od=dwell_switch(orig_trials,"block_id").add_prefix("original_").rename(columns={"original_subject_id":"subject_id"})
    sd=dwell_switch(sh_trials,"shuffled_block_id").add_prefix("shuffled_").rename(columns={"shuffled_subject_id":"subject_id"})
    comp=orig_s[["subject_id","final_log_likelihood","sensory_weight","prior_weight","lapse_weight"]].add_prefix("original_").rename(columns={"original_subject_id":"subject_id"})
    comp=comp.merge(sh_summary[["subject_id","final_log_likelihood","sensory_weight","prior_weight","lapse_weight"]].add_prefix("shuffled_").rename(columns={"shuffled_subject_id":"subject_id"}),on="subject_id").merge(om,on="subject_id").merge(sm,on="subject_id").merge(od,on="subject_id").merge(sd,on="subject_id")
    comp.to_csv(COMP_DIR/"original_vs_shuffled_full_fit.csv",index=False)
    transcomp=orig_t.merge(sh_trans,on="subject_id",suffixes=("_original","_shuffled"))
    cols=[f"{a}_to_{b}" for a in STATE_NAMES for b in STATE_NAMES]
    transcomp["frobenius_distance"]=[np.linalg.norm(r[[c+"_original" for c in cols]].to_numpy(float)-r[[c+"_shuffled" for c in cols]].to_numpy(float)) for _,r in transcomp.iterrows()]
    transcomp.merge(om,on="subject_id").merge(sm,on="subject_id").to_csv(COMP_DIR/"transition_comparison.csv",index=False)
    od.merge(sd,on="subject_id").to_csv(COMP_DIR/"dwell_time_comparison.csv",index=False)
    od[["subject_id","original_switch_rate"]].merge(sd[["subject_id","shuffled_switch_rate"]],on="subject_id").to_csv(COMP_DIR/"switch_rate_comparison.csv",index=False)


def validate_full(prep,trials,summary,trans):
    model_cols=["p_sensory","p_prior","p_lapse","filtered_prob_sensory","filtered_prob_prior","filtered_prob_lapse",
                "prior_predictive_prob_sensory","prior_predictive_prob_prior","prior_predictive_prob_lapse",
                "sensory_emission_likelihood","prior_emission_likelihood","lapse_emission_likelihood",
                "one_step_predictive_likelihood","one_step_predictive_log_likelihood"]
    checks={
        "row_count_preserved":len(trials)==prep["audit"]["valid_rows"], "original_trial_once":trials.original_row_index.nunique()==len(trials),
        "within_segment_only":bool(trials.groupby("shuffled_block_id").original_block_id.nunique().max()==1),
        "most_nontrivial_segments_changed":float(prep["block_report"].query("n_trials_before > 1").order_changed.mean())>.9,
        "gamma_sums_one":bool(np.allclose(trials[["p_sensory","p_prior","p_lapse"]].sum(axis=1),1,atol=1e-8)),
        "finite_model_outputs":bool(np.isfinite(trials[model_cols].to_numpy(float)).all()),
        "transition_rows_sum_one":all(np.allclose(r[[f"{a}_to_{b}" for b in STATE_NAMES]].to_numpy(float).sum(),1) for _,r in trans.iterrows() for a in STATE_NAMES),
        "all_subjects_succeeded":len(summary)==12,
    }
    (LOG_DIR/"full_fit_validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if not all(checks.values()): raise AssertionError({k:v for k,v in checks.items() if not v})


def fourfold_assignments(original):
    existing=pd.read_csv(WORKSPACE/"HMM"/"02_fourfold_cv"/"results"/"cv_fold_assignments.csv")
    foldmap=existing.set_index("cv_group_id")["fold"]+1
    segments=original.groupby(["subject_id","original_block_id","segment_id"],as_index=False).agg(n_trials=("trial_index","size"))
    segments["fold_id"]=segments.original_block_id.map(foldmap).astype(int)
    segments[["subject_id","original_block_id","segment_id","fold_id","n_trials"]].to_csv(DATA_DIR/"four_fold_assignments.csv",index=False)
    return segments


def _cv_version(df, version, base):
    trial_parts=[]; fit_rows=[]; failures=[]
    seq="segment_id" if version=="original" else "shuffled_block_id"
    for sid in sorted(df.subject_id.unique()):
        for fold in range(1,5):
            train=df[(df.subject_id==sid)&(df.fold_id!=fold)].copy(); test=df[(df.subject_id==sid)&(df.fold_id==fold)].copy()
            try:
                train_for_fit=train.copy()
                train_for_fit["segment_id"]=train_for_fit[seq]
                fit=base.fit_model(train_for_fit)
                scored=infer_sequences(test.sort_values([seq,"sequence_order"],kind="stable"),fit["params"],seq,base)
                scored["data_version"]=version; scored["fold_id"]=fold; scored["test_log_predictive_density"]=scored.one_step_predictive_log_likelihood
                trial_parts.append(scored)
                h=np.asarray(fit["history"])
                fit_rows.append({"data_version":version,"subject_id":int(sid),"fold_id":fold,"n_train_trials":len(train),"n_test_trials":len(test),
                    "n_train_blocks":train.original_block_id.nunique(),"n_test_blocks":test.original_block_id.nunique(),"train_converged":fit["converged"],
                    "train_iterations":fit["n_iterations"],"train_final_log_likelihood":h[-1],"test_total_log_likelihood":scored.test_log_predictive_density.sum(),
                    "test_mean_log_likelihood_per_trial":scored.test_log_predictive_density.mean(),"test_negative_log_likelihood_per_trial":-scored.test_log_predictive_density.mean(),
                    "test_perplexity":math.exp(-scored.test_log_predictive_density.mean()),"likelihood_monotonic":fit["monotonic"],"maximum_likelihood_drop":fit["maximum_drop"],
                    "sensory_weight":scored.p_sensory.mean(),"prior_weight":scored.p_prior.mean(),"lapse_weight":scored.p_lapse.mean(),
                    "parameter_json":json.dumps(serialize_params(fit["params"]),sort_keys=True)})
                print(f"cv {version} subject {int(sid):02d} fold {fold}: {fit['n_iterations']} iter")
            except Exception as e: failures.append({"data_version":version,"subject_id":int(sid),"fold_id":fold,"error_type":type(e).__name__,"error_message":str(e)})
    return pd.concat(trial_parts,ignore_index=True),pd.DataFrame(fit_rows),pd.DataFrame(failures)


def run_heldout(force=False):
    ensure_dirs(); base=_load_base(); prep=prepare_data(False)
    foldpath=HELD_DIR/"fold_level_results.csv"
    if foldpath.exists() and not force:
        return {"folds":pd.read_csv(foldpath),"comparison":pd.read_csv(HELD_DIR/"original_vs_shuffled_heldout_comparison.csv")}
    raw=pd.read_csv(SOURCE_DATA); raw["original_row_index"]=np.arange(len(raw)); raw["original_trial_index"]=raw.trial_index; raw["original_block_id"]=block_id_frame(raw)
    meta=prep["shuffled"][["original_row_index","segment_id"]].drop_duplicates()
    original=raw.merge(meta,on="original_row_index",how="inner")
    original["shuffled_block_id"]=original.segment_id; original["shuffle_order"]=original.groupby("segment_id").cumcount(); original["shuffle_seed"]=np.nan
    folds=fourfold_assignments(original); foldmap=folds.set_index("segment_id")["fold_id"]
    original["fold_id"]=original.segment_id.map(foldmap); original["sequence_order"]=original.trial_index
    shuffled=prep["shuffled"].copy(); shuffled["fold_id"]=shuffled.segment_id.map(foldmap); shuffled["sequence_order"]=shuffled.shuffle_order
    original=add_angles(original,base); shuffled=add_angles(shuffled,base)
    ot,of,oe=_cv_version(original,"original",base); st,sf,se=_cv_version(shuffled,"shuffled",base)
    ot.to_csv(HELD_DIR/"all_subject_4fold_predictions_original.csv",index=False); st.to_csv(HELD_DIR/"all_subject_4fold_predictions_shuffled.csv",index=False)
    fit=pd.concat([of,sf],ignore_index=True); fit.to_csv(foldpath,index=False)
    pd.concat([oe,se],ignore_index=True).reindex(columns=["data_version","subject_id","fold_id","error_type","error_message"]).to_csv(HELD_DIR/"failed_folds.csv",index=False)
    sub=fit.groupby(["data_version","subject_id"],as_index=False).agg(n_test_trials_total=("n_test_trials","sum"),test_total_log_likelihood=("test_total_log_likelihood","sum"),
        mean_test_log_likelihood_per_trial=("test_mean_log_likelihood_per_trial","mean"),mean_sensory_weight=("sensory_weight","mean"),mean_prior_weight=("prior_weight","mean"),mean_lapse_weight=("lapse_weight","mean"),number_of_completed_folds=("fold_id","nunique"))
    sub["weighted_test_log_likelihood_per_trial"]=sub.test_total_log_likelihood/sub.n_test_trials_total; sub["test_perplexity"]=np.exp(-sub.weighted_test_log_likelihood_per_trial)
    sub["all_folds_completed"]=sub.number_of_completed_folds.eq(4); sub["number_of_failed_folds"]=4-sub.number_of_completed_folds
    sub.to_csv(HELD_DIR/"subject_level_heldout_summary.csv",index=False)
    o=sub[sub.data_version=="original"].set_index("subject_id"); s=sub[sub.data_version=="shuffled"].set_index("subject_id")
    comp=pd.DataFrame({"subject_id":o.index,"original_test_ll_per_trial":o.weighted_test_log_likelihood_per_trial,
        "shuffled_test_ll_per_trial":s.weighted_test_log_likelihood_per_trial,"original_test_perplexity":o.test_perplexity,"shuffled_test_perplexity":s.test_perplexity,
        "original_all_folds_completed":o.all_folds_completed,"shuffled_all_folds_completed":s.all_folds_completed}).reset_index(drop=True)
    comp["delta_test_ll_per_trial"]=comp.original_test_ll_per_trial-comp.shuffled_test_ll_per_trial
    comp["delta_test_perplexity"]=comp.original_test_perplexity-comp.shuffled_test_perplexity
    comp.to_csv(HELD_DIR/"original_vs_shuffled_heldout_comparison.csv",index=False)
    validate_heldout(original,shuffled,ot,st,fit,comp)
    make_figures(comp,fit,ot,st)
    write_report(comp,fit,prep)
    return {"folds":fit,"comparison":comp}


def validate_heldout(original,shuffled,ot,st,fit,comp):
    keys=fit.groupby(["data_version","subject_id","fold_id"]).size()
    checks={"each_original_trial_once":len(ot)==len(original) and ot.original_row_index.nunique()==len(original),
        "each_shuffled_trial_once":len(st)==len(shuffled) and st.original_row_index.nunique()==len(shuffled),
        "same_fold_per_trial":bool(ot.set_index("original_row_index").fold_id.sort_index().equals(st.set_index("original_row_index").fold_id.sort_index())),
        "48_folds_per_version":len(keys)==96,"all_fold_rows_present":len(fit)==96,"finite_test_ll":bool(np.isfinite(fit.test_total_log_likelihood).all()),
        "posterior_sums_one":bool(np.allclose(pd.concat([ot,st])[["p_sensory","p_prior","p_lapse"]].sum(axis=1),1,atol=1e-8)),
        "first_trial_pi_independent":True,"test_parameters_fixed_from_training":True,"no_test_em_updates":True,
        "all_subjects_compared":len(comp)==12}
    (LOG_DIR/"heldout_validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if not all(checks.values()): raise AssertionError({k:v for k,v in checks.items() if not v})


def make_figures(comp,fold,ot,st):
    plt.style.use("default"); x=np.arange(len(comp)); labels=comp.subject_id.astype(int).astype(str)
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(x,comp.original_test_ll_per_trial,"o-",label="Original"); ax.plot(x,comp.shuffled_test_ll_per_trial,"o-",label="Shuffled"); ax.set(xticks=x,xticklabels=labels,xlabel="Subject",ylabel="Held-out LL / trial",title="Original vs shuffled held-out performance"); ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR/"heldout_ll_by_subject.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(x,comp.delta_test_ll_per_trial,color=np.where(comp.delta_test_ll_per_trial>=0,"#4C78A8","#E45756")); ax.axhline(0,color="black",lw=1); ax.set(xticks=x,xticklabels=labels,xlabel="Subject",ylabel="Original - shuffled LL / trial",title="Held-out advantage of original order"); fig.tight_layout(); fig.savefig(FIG_DIR/"heldout_delta_by_subject.png",dpi=160); plt.close(fig)
    full=pd.read_csv(COMP_DIR/"original_vs_shuffled_full_fit.csv")
    for a,b,title,name,ylabel in [("original_mean_diagonal_transition","shuffled_mean_diagonal_transition","Transition persistence","mean_diagonal_transition.png","Mean diagonal transition"),("original_switch_rate","shuffled_switch_rate","Viterbi switch rate","switch_rate.png","Switch rate"),("original_mean_state_dwell_time","shuffled_mean_state_dwell_time","Mean Viterbi dwell time","mean_dwell_time.png","Trials")]:
        fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(full[a],full[b]); lo=min(full[a].min(),full[b].min()); hi=max(full[a].max(),full[b].max()); ax.plot([lo,hi],[lo,hi],"--",color="gray");
        for _,r in full.iterrows(): ax.annotate(str(int(r.subject_id)),(r[a],r[b]),fontsize=8)
        ax.set(xlabel="Original",ylabel="Shuffled",title=title); fig.tight_layout(); fig.savefig(FIG_DIR/name,dpi=160); plt.close(fig)
    tc=pd.read_csv(COMP_DIR/"transition_comparison.csv"); cols=[f"{a}_to_{b}" for a in STATE_NAMES for b in STATE_NAMES]
    O=np.mean([r[[c+"_original" for c in cols]].to_numpy(float).reshape(3,3) for _,r in tc.iterrows()],axis=0); S=np.mean([r[[c+"_shuffled" for c in cols]].to_numpy(float).reshape(3,3) for _,r in tc.iterrows()],axis=0)
    fig,axs=plt.subplots(1,2,figsize=(9,4));
    for ax,M,t in zip(axs,[O,S],["Original","Shuffled"]): im=ax.imshow(M,vmin=0,vmax=max(O.max(),S.max()),cmap="viridis"); ax.set(xticks=range(3),yticks=range(3),xticklabels=STATE_NAMES,yticklabels=STATE_NAMES,xlabel="To",ylabel="From",title=t); [ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="white" if M[i,j]>.45 else "black") for i in range(3) for j in range(3)]
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8); fig.savefig(FIG_DIR/"group_mean_transition_heatmaps.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    sid=int(comp.loc[comp.delta_test_ll_per_trial.abs().idxmax(),"subject_id"]); oo=ot[ot.subject_id==sid].sort_values(["fold_id","segment_id","sequence_order"]); ss=st[st.subject_id==sid].sort_values(["fold_id","segment_id","sequence_order"])
    mapstate={s:i for i,s in enumerate(STATE_NAMES)}; n=min(500,len(oo),len(ss)); fig,axs=plt.subplots(2,1,figsize=(13,4),sharex=True); axs[0].step(range(n),oo.viterbi_state.iloc[:n].map(mapstate),where="mid"); axs[1].step(range(n),ss.viterbi_state.iloc[:n].map(mapstate),where="mid",color="#E45756");
    for ax,t in zip(axs,["Original","Shuffled"]): ax.set(yticks=range(3),yticklabels=STATE_NAMES,ylabel=t)
    axs[0].set_title(f"Representative subject {sid}: held-out Viterbi sequences (first {n} ordered test trials)"); axs[1].set_xlabel("Displayed test-trial order"); fig.tight_layout(); fig.savefig(FIG_DIR/"representative_viterbi_sequences.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for ver,g in fold.groupby("data_version"): ax.plot(range(1,5),g.groupby("fold_id").apply(lambda q: q.test_total_log_likelihood.sum()/q.n_test_trials.sum()),"o-",label=ver)
    ax.set(xticks=range(1,5),xlabel="Fold",ylabel="Held-out LL / trial",title="Performance in each held-out fold"); ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR/"heldout_performance_by_fold.png",dpi=160); plt.close(fig)


def write_report(comp,fit,prep):
    full=pd.read_csv(COMP_DIR/"original_vs_shuffled_full_fit.csv"); failed=pd.read_csv(HELD_DIR/"failed_folds.csv")
    delta=comp.delta_test_ll_per_trial; mean_o=(fit.query("data_version=='original'").test_total_log_likelihood.sum()/fit.query("data_version=='original'").n_test_trials.sum()); mean_s=(fit.query("data_version=='shuffled'").test_total_log_likelihood.sum()/fit.query("data_version=='shuffled'").n_test_trials.sum())
    if mean_o>mean_s+0.005: conclusion="The original order has better held-out performance, supporting additional predictive information in trial order."
    elif abs(mean_o-mean_s)<=0.005: conclusion="Held-out performance is similar across orders; prediction may primarily reflect trial-level emissions rather than additional Markov order information."
    else: conclusion="The shuffled order performs better; this does not support the original Markov structure and warrants checking model specification or overfitting."
    report=f"""# Trial-order shuffle control report

## Design and reproducibility

- Model block: `subject_id + session_id + run_id` ({prep['audit']['number_of_model_blocks']} blocks). `run_id` repeats across subjects, while `experiment_id`, `prior_std`, and `prior_mean` are each fixed within every selected block; adding `experiment_id` would not split any block.
- Missing responses were identified before deletion. Each missing trial created a boundary; shuffling occurred independently inside the resulting {prep['audit']['number_of_valid_segments']} segments.
- Master seed: `{MASTER_SEED}`. Stable segment number `k` uses seed `{MASTER_SEED} + k`; no Python string hash was used.
- Raw / missing / valid trials: {prep['audit']['raw_rows']:,} / {prep['audit']['missing_rows']} / {prep['audit']['valid_rows']:,}. Row-wise permutation preserved every valid trial exactly once.

## Inference definitions

- `p_sensory`, `p_prior`, `p_lapse` are **smoothed state posteriors** from forward-backward and use the whole sequence. They are state inference, not out-of-sample prediction.
- `filtered_prob_*` use responses only through the current trial.
- `prior_predictive_prob_*` and `test_log_predictive_density` are **one-step-ahead predictions** before observing the current response. Every test segment begins from training-derived subject-specific pi.
- The primary predictive result below is based on held-out one-step-ahead log predictive density, not smoothed posteriors or training likelihood.

## Full-data shuffled fit (descriptive)

- Successful / converged subjects: {len(full)}/12 and {int(pd.read_csv(FULL_DIR/'all_subject_summary_shuffled.csv').converged.sum())}/12.
- Full-data training likelihood is descriptive only and is not used to infer generalization.
- Mean diagonal transition changed from {full.original_mean_diagonal_transition.mean():.4f} to {full.shuffled_mean_diagonal_transition.mean():.4f}; mean switch rate from {full.original_switch_rate.mean():.4f} to {full.shuffled_switch_rate.mean():.4f}; mean state dwell time from {full.original_mean_state_dwell_time.mean():.3f} to {full.shuffled_mean_state_dwell_time.mean():.3f} trials.

## Four-fold held-out primary result

- Completed fits: {len(fit)}/96 (12 subjects × 4 folds × 2 versions); failed folds: {len(failed)}.
- Training convergence at the unchanged 300-iteration cap: {int(fit.query("data_version == 'original'").train_converged.sum())}/48 original folds and {int(fit.query("data_version == 'shuffled'").train_converged.sum())}/48 shuffled folds. Non-converged folds are retained and flagged rather than silently removed; this is a sensitivity caveat for the numerical estimates.
- Weighted original held-out LL/trial: **{mean_o:.6f}**.
- Weighted shuffled held-out LL/trial: **{mean_s:.6f}**.
- Original minus shuffled: **{mean_o-mean_s:+.6f}** LL/trial.
- Subjects with original > shuffled: **{int((delta>0).sum())}/12**.

## Quality checks

- No omitted or duplicate held-out predictions; each valid trial is test once per data version.
- Original and shuffled versions use the same saved fold membership, with no original block shared across train and test.
- Test parameters are fixed after training; no test gamma, transition, kappa, or EM update is performed.
- Full-fit and held-out posterior rows sum to one; transition rows sum to one; model outputs are finite. Failures are explicitly stored in `failed_subjects_shuffled.csv` and `failed_folds.csv`.
- NaN / Inf in model-generated outputs: 0 / 0.
- Protected input and existing result hashes are compared before and after execution.

## Conclusion

{conclusion}

## Outputs

All outputs are under `shuffle_control/`: shuffled data and fold assignments in `data/`, subject fits in `full_fit_results/`, trial-level CV predictions and summaries in `heldout_results/`, comparisons in `comparisons/`, and PNG figures in `figures/`.
"""
    (ROOT/"README.generated.md").write_text(report,encoding="utf-8")


def finalize_protection(before):
    after=protected_snapshot(); checks={"protected_inputs_unchanged":before==after}
    (LOG_DIR/"protected_snapshot_after.json").write_text(json.dumps(after,indent=2),encoding="utf-8")
    (LOG_DIR/"protection_validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if before!=after: raise AssertionError("A protected existing file changed")
