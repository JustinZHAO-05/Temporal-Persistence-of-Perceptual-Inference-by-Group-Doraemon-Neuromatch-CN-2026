from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "exchangeable_block_mixture"
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
COMP_DIR = ROOT / "comparisons"
FIG_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"
MASTER_SEED = 20260717
MAX_ITER = 300
TOL = 1e-6
SMOOTHING = 1e-6
STATE_NAMES = np.array(["sensory", "prior", "lapse"])
MODEL_NAMES = ["subject_iid", "prior_conditioned_iid", "exchangeable_block_mixture"]


def base_module():
    core = WORKSPACE / "hybrid" / "common"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    import fourfold_cv_core as base
    return base


def ensure_dirs():
    for p in (DATA_DIR, RESULT_DIR, COMP_DIR, FIG_DIR, LOG_DIR, ROOT / "model_artifacts"):
        p.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_snapshot():
    paths = [
        WORKSPACE / "HMM" / "01_full_data_fit" / "data" / "data01_direction4priors.csv",
        WORKSPACE / "HMM" / "01_full_data_fit" / "results" / "all_subject_summary_revised.csv",
        WORKSPACE / "HMM" / "03_shuffle_control" / "results" / "heldout" / "fold_level_results.csv",
        WORKSPACE / "HMM" / "03_shuffle_control" / "results" / "heldout" / "all_subject_4fold_predictions_original.csv",
        WORKSPACE / "HMM" / "03_shuffle_control" / "results" / "heldout" / "all_subject_4fold_predictions_shuffled.csv",
    ]
    return {str(p): {"size": p.stat().st_size, "sha256": sha256(p)} for p in paths}


def load_data():
    original = pd.read_csv(WORKSPACE / "HMM" / "03_shuffle_control" / "results" / "heldout" / "all_subject_4fold_predictions_original.csv")
    shuffled = pd.read_csv(WORKSPACE / "HMM" / "03_shuffle_control" / "results" / "heldout" / "all_subject_4fold_predictions_shuffled.csv")
    keep = [
        "subject_id", "experiment_id", "session_id", "run_id", "original_block_id", "segment_id",
        "shuffled_block_id", "original_row_index", "original_trial_index", "shuffle_order", "shuffle_seed",
        "motion_direction", "motion_coherence", "prior_mean", "prior_std", "estimate_x", "estimate_y",
        "response_direction", "x_rad", "y_rad", "fold_id", "sequence_order",
    ]
    original = original[keep].copy()
    shuffled = shuffled[keep].copy()
    original["data_version"] = "original"
    shuffled["data_version"] = "shuffled"
    if len(original) != 83210 or len(shuffled) != 83210:
        raise AssertionError("Unexpected valid trial count")
    a = original.set_index("original_row_index")["fold_id"].sort_index()
    b = shuffled.set_index("original_row_index")["fold_id"].sort_index()
    if not a.equals(b):
        raise AssertionError("Original and shuffled folds differ")
    audit = pd.DataFrame([
        {"data_version": "original", "n_trials": len(original), "n_subjects": original.subject_id.nunique(),
         "n_blocks": original.original_block_id.nunique(), "n_segments": original.segment_id.nunique(),
         "n_folds": original.fold_id.nunique(), "duplicate_trials": original.original_row_index.duplicated().sum()},
        {"data_version": "shuffled", "n_trials": len(shuffled), "n_subjects": shuffled.subject_id.nunique(),
         "n_blocks": shuffled.original_block_id.nunique(), "n_segments": shuffled.segment_id.nunique(),
         "n_folds": shuffled.fold_id.nunique(), "duplicate_trials": shuffled.original_row_index.duplicated().sum()},
    ])
    audit.to_csv(DATA_DIR / "dataset_audit.csv", index=False)
    original[["subject_id", "original_block_id", "segment_id", "fold_id"]].drop_duplicates().to_csv(DATA_DIR / "four_fold_assignments_verified.csv", index=False)
    return original, shuffled


def initial_params():
    return {
        "weights": np.array([0.60, 0.35, 0.05]),
        "kappaS": {0.06: 1.5, 0.12: 4.5, 0.24: 20.0},
        "kappaP": {80: 0.5, 40: 1.0, 20: 6.0, 10: 30.0},
    }


def update_kappas(df, gamma, params, base):
    out = {"kappaS": params["kappaS"].copy(), "kappaP": params["kappaP"].copy()}
    sensory_error = base.wrap_angle_rad(df.y_rad.to_numpy(float) - df.x_rad.to_numpy(float))
    prior_error = base.wrap_angle_rad(df.y_rad.to_numpy(float))
    coherence = df.motion_coherence.to_numpy(float)
    prior_std = df.prior_std.to_numpy(int)
    for level in (0.06, 0.12, 0.24):
        mask = coherence == level; w = gamma[mask, 0]
        resultant = np.sum(w * np.cos(sensory_error[mask])) / max(w.sum(), 1e-15)
        out["kappaS"][level] = base.concentration_from_resultant(float(resultant))
    for level in (10, 20, 40, 80):
        mask = prior_std == level; w = gamma[mask, 1]
        resultant = np.sum(w * np.cos(prior_error[mask])) / max(w.sum(), 1e-15)
        out["kappaP"][level] = base.concentration_from_resultant(float(resultant))
    return out


def emission(df, params, base):
    bridge = {"initial_prob": np.array([.6,.35,.05]), "transition_matrix": np.eye(3),
              "kappaS": params["kappaS"], "kappaP": params["kappaP"]}
    return base.log_emission_matrix(df, bridge)


def finish(history, params, extra=None):
    d = np.diff(history)
    return {"params": params, "history": history, "converged": len(history)>1 and abs(history[-1]-history[-2])<TOL,
            "n_iterations": len(history), "monotonic": bool(len(d)==0 or d.min()>=-1e-8),
            "maximum_drop": float(max(0, -d.min())) if len(d) else 0, **(extra or {})}


def fit_subject_iid(df, base):
    p = initial_params(); history=[]
    for it in range(MAX_ITER):
        le = emission(df,p,base); joint = le + np.log(np.clip(p["weights"],1e-300,None)); norm=logsumexp(joint,axis=1)
        gamma=np.exp(joint-norm[:,None]); history.append(float(norm.sum()))
        if it and abs(history[-1]-history[-2])<TOL: break
        p["weights"]=(gamma.sum(axis=0)+SMOOTHING)/(len(df)+3*SMOOTHING)
        kp=update_kappas(df,gamma,p,base);p.update(kp)
    return finish(history,p)


def fit_prior_conditioned(df, base):
    p=initial_params(); p["condition_weights"]={level:p["weights"].copy() for level in (10,20,40,80)}; history=[]
    cond=df.prior_std.to_numpy(int)
    for it in range(MAX_ITER):
        le=emission(df,p,base); W=np.vstack([p["condition_weights"][int(c)] for c in cond]); joint=le+np.log(np.clip(W,1e-300,None)); norm=logsumexp(joint,axis=1)
        gamma=np.exp(joint-norm[:,None]);history.append(float(norm.sum()))
        if it and abs(history[-1]-history[-2])<TOL:break
        for level in (10,20,40,80):
            g=gamma[cond==level].sum(axis=0)+SMOOTHING;p["condition_weights"][level]=g/g.sum()
        kp=update_kappas(df,gamma,p,base);p.update(kp)
    return finish(history,p)


def block_expectation(df,p,base):
    le=emission(df,p,base); C=len(p["rho"]); gamma=np.zeros((len(df),3)); class_counts=np.zeros(C); weight_counts=np.zeros((C,3)); ll=0.; taus={}
    logW=np.log(np.clip(p["class_weights"],1e-300,None)); logr=np.log(np.clip(p["rho"],1e-300,None))
    for seg,g in df.groupby("segment_id",sort=False):
        pos=g.index.to_numpy(); L=le[pos]; joint=L[:,None,:]+logW[None,:,:]; logmix=logsumexp(joint,axis=2)
        logclass=logr+logmix.sum(axis=0); norm=logsumexp(logclass); tau=np.exp(logclass-norm); resp=np.exp(joint-logmix[:,:,None])
        gamma[pos]=np.einsum("c,tck->tk",tau,resp);class_counts+=tau;weight_counts+=tau[:,None]*resp.sum(axis=0);ll+=float(norm);taus[str(seg)]=tau
    return ll,gamma,class_counts,weight_counts,taus


def fit_block_mixture(df,base):
    p=initial_params();p["rho"]=np.array([1/3]*3);p["class_weights"]=np.array([[.88,.10,.02],[.10,.88,.02],[.47,.43,.10]]);history=[]
    local=df.reset_index(drop=True)
    for it in range(MAX_ITER):
        ll,gamma,cc,wc,taus=block_expectation(local,p,base);history.append(ll)
        if it and abs(history[-1]-history[-2])<TOL:break
        p["rho"]=(cc+SMOOTHING)/(cc.sum()+3*SMOOTHING);p["class_weights"]=(wc+SMOOTHING)/(wc.sum(axis=1,keepdims=True)+3*SMOOTHING)
        kp=update_kappas(local,gamma,p,base);p.update(kp)
    return finish(history,p,{"training_block_posteriors":taus})


def fit_model(name,df,base):
    if name=="subject_iid":return fit_subject_iid(df,base)
    if name=="prior_conditioned_iid":return fit_prior_conditioned(df,base)
    if name=="exchangeable_block_mixture":return fit_block_mixture(df,base)
    raise ValueError(name)


def score_model(name,df,p,base):
    local=df.sort_values(["segment_id","sequence_order"],kind="stable").reset_index(drop=True).copy();le=emission(local,p,base);E=np.exp(le)
    q=np.zeros((len(local),3));post=np.zeros_like(q);density=np.zeros(len(local))
    if name=="subject_iid":
        q[:]=p["weights"]
    elif name=="prior_conditioned_iid":
        q=np.vstack([p["condition_weights"][int(c)] for c in local.prior_std])
    else:
        for seg,g in local.groupby("segment_id",sort=False):
            tau=p["rho"].copy()
            for pos in g.index:
                q[pos]=tau@p["class_weights"]
                class_like=p["class_weights"]@E[pos]
                tau=tau*class_like;tau/=tau.sum()
    density=(q*E).sum(axis=1);post=q*E/density[:,None]
    local[["prior_predictive_prob_sensory","prior_predictive_prob_prior","prior_predictive_prob_lapse"]]=q
    local[["p_sensory","p_prior","p_lapse"]]=post
    local["most_likely_state"]=STATE_NAMES[np.argmax(post,axis=1)]
    local[["sensory_emission_likelihood","prior_emission_likelihood","lapse_emission_likelihood"]]=E
    local["test_predictive_likelihood"]=density;local["test_log_predictive_density"]=np.log(density);local["model_name"]=name
    return local


def jsonable(p):
    out={"kappaS":{str(k):float(v) for k,v in p["kappaS"].items()},"kappaP":{str(k):float(v) for k,v in p["kappaP"].items()}}
    for k in ("weights","rho","class_weights"):
        if k in p:out[k]=np.asarray(p[k]).tolist()
    if "condition_weights" in p:out["condition_weights"]={str(k):np.asarray(v).tolist() for k,v in p["condition_weights"].items()}
    return out


def run_full_fits(original,base):
    rows=[]
    for sid in sorted(original.subject_id.unique()):
        df=original[original.subject_id==sid].reset_index(drop=True)
        for name in MODEL_NAMES:
            fit=fit_model(name,df,base);h=np.asarray(fit["history"]);out=RESULT_DIR/"full_fit"/name/f"subject_{int(sid):02d}";out.mkdir(parents=True,exist_ok=True)
            (out/"parameters.json").write_text(json.dumps(jsonable(fit["params"]),indent=2),encoding="utf-8")
            pd.DataFrame({"iteration":np.arange(1,len(h)+1),"log_likelihood":h}).to_csv(out/"log_likelihood_history.csv",index=False)
            rows.append({"model_name":name,"subject_id":int(sid),"n_trials":len(df),"converged":fit["converged"],"n_iterations":fit["n_iterations"],"final_log_likelihood":h[-1],"ll_per_trial":h[-1]/len(df),"monotonic":fit["monotonic"],"parameter_json":json.dumps(jsonable(fit["params"]),sort_keys=True)})
            print(f"full {name} subject {int(sid):02d}: iter={fit['n_iterations']}")
    out=pd.DataFrame(rows);out.to_csv(RESULT_DIR/"full_fit_summary.csv",index=False);return out


def run_cv(original,shuffled,base):
    folds=[];pred_parts={};fit_cache={}
    for sid in sorted(original.subject_id.unique()):
        for fold in range(1,5):
            train=original[(original.subject_id==sid)&(original.fold_id!=fold)].reset_index(drop=True)
            for name in MODEL_NAMES:
                start=time.perf_counter();fit=fit_model(name,train,base);fit_cache[(name,int(sid),fold)]=fit
                for version,data in (("original",original),("shuffled",shuffled)):
                    test=data[(data.subject_id==sid)&(data.fold_id==fold)].copy();scored=score_model(name,test,fit["params"],base);scored["data_version"]=version;scored["fold_id"]=fold
                    pred_parts.setdefault((name,version),[]).append(scored)
                    h=np.asarray(fit["history"]);folds.append({"model_name":name,"data_version":version,"subject_id":int(sid),"fold_id":fold,
                        "n_train_trials":len(train),"n_test_trials":len(scored),"n_train_blocks":train.original_block_id.nunique(),"n_test_blocks":scored.original_block_id.nunique(),
                        "train_converged":fit["converged"],"train_iterations":fit["n_iterations"],"train_final_log_likelihood":h[-1],
                        "test_total_log_likelihood":scored.test_log_predictive_density.sum(),"test_ll_per_trial":scored.test_log_predictive_density.mean(),
                        "test_perplexity":math.exp(-scored.test_log_predictive_density.mean()),"monotonic":fit["monotonic"],"maximum_drop":fit["maximum_drop"],
                        "runtime_seconds":time.perf_counter()-start,"parameter_json":json.dumps(jsonable(fit["params"]),sort_keys=True)})
                print(f"cv {name} subject {int(sid):02d} fold {fold}: iter={fit['n_iterations']}")
    cols=["model_name","data_version","subject_id","fold_id","original_block_id","segment_id","original_row_index","original_trial_index","sequence_order","shuffle_order",
          "motion_direction","motion_coherence","prior_mean","prior_std","response_direction","x_rad","y_rad","prior_predictive_prob_sensory","prior_predictive_prob_prior","prior_predictive_prob_lapse",
          "p_sensory","p_prior","p_lapse","most_likely_state","sensory_emission_likelihood","prior_emission_likelihood","lapse_emission_likelihood","test_predictive_likelihood","test_log_predictive_density"]
    for key,parts in pred_parts.items():
        p=pd.concat(parts,ignore_index=True);p[cols].to_csv(RESULT_DIR/f"heldout_predictions_{key[0]}_{key[1]}.csv",index=False)
    fold_df=pd.DataFrame(folds);fold_df.to_csv(RESULT_DIR/"fold_level_results.csv",index=False);return fold_df,pred_parts


def summaries(fold_df):
    subject=(fold_df.groupby(["model_name","data_version","subject_id"],as_index=False).agg(n_test_trials=("n_test_trials","sum"),total_ll=("test_total_log_likelihood","sum"),completed_folds=("fold_id","nunique"),converged_folds=("train_converged","sum")))
    subject["ll_per_trial"]=subject.total_ll/subject.n_test_trials;subject["perplexity"]=np.exp(-subject.ll_per_trial);subject.to_csv(RESULT_DIR/"subject_level_summary.csv",index=False)
    overall=(subject.groupby(["model_name","data_version"],as_index=False).agg(n_test_trials=("n_test_trials","sum"),total_ll=("total_ll","sum"),subjects=("subject_id","nunique"),completed_folds=("completed_folds","sum"),converged_folds=("converged_folds","sum")))
    overall["ll_per_trial"]=overall.total_ll/overall.n_test_trials;overall["perplexity"]=np.exp(-overall.ll_per_trial)
    hmm=pd.read_csv(WORKSPACE/"HMM"/"03_shuffle_control"/"results"/"heldout"/"fold_level_results.csv")
    ho=(hmm.groupby("data_version",as_index=False).agg(n_test_trials=("n_test_trials","sum"),total_ll=("test_total_log_likelihood","sum"),converged_folds=("train_converged","sum")))
    ho["model_name"]="hmm";ho["subjects"]=12;ho["completed_folds"]=48;ho["ll_per_trial"]=ho.total_ll/ho.n_test_trials;ho["perplexity"]=np.exp(-ho.ll_per_trial)
    overall=pd.concat([overall,ho[overall.columns]],ignore_index=True);overall.to_csv(COMP_DIR/"overall_model_comparison.csv",index=False)
    hs=(hmm.groupby(["data_version","subject_id"],as_index=False).agg(n_test_trials=("n_test_trials","sum"),total_ll=("test_total_log_likelihood","sum")));hs["ll_per_trial"]=hs.total_ll/hs.n_test_trials;hs["model_name"]="hmm"
    allsub=pd.concat([subject[["model_name","data_version","subject_id","n_test_trials","total_ll","ll_per_trial"]],hs],ignore_index=True);allsub.to_csv(COMP_DIR/"all_model_subject_comparison.csv",index=False)
    wide=overall.pivot(index="model_name",columns="data_version",values="ll_per_trial")
    baseline=wide.loc["subject_iid","original"];condition=wide.loc["prior_conditioned_iid","original"];block=wide.loc["exchangeable_block_mixture","original"]
    rows=[]
    for bname,bval in [("subject_iid",baseline),("prior_conditioned_iid",condition),("exchangeable_block_mixture",block)]:
        go=wide.loc["hmm","original"]-bval;gs=wide.loc["hmm","shuffled"]-bval
        rows.append({"baseline":bname,"baseline_ll_per_trial":bval,"hmm_original_gain":go,"hmm_shuffled_gain":gs,"order_specific_difference_in_differences":go-gs})
    decomp=pd.DataFrame(rows);decomp.to_csv(COMP_DIR/"markov_gain_decomposition.csv",index=False)
    invariance=[]
    for name in MODEL_NAMES:
        a=fold_df[(fold_df.model_name==name)&(fold_df.data_version=="original")].sort_values(["subject_id","fold_id"])
        b=fold_df[(fold_df.model_name==name)&(fold_df.data_version=="shuffled")].sort_values(["subject_id","fold_id"])
        diff=a.test_total_log_likelihood.to_numpy()-b.test_total_log_likelihood.to_numpy()
        invariance.append({"model_name":name,"max_abs_fold_total_ll_difference":np.max(np.abs(diff)),"overall_total_ll_difference":diff.sum(),"order_invariant_pass":np.max(np.abs(diff))<1e-7})
    pd.DataFrame(invariance).to_csv(COMP_DIR/"iid_order_invariance.csv",index=False)
    return subject,overall,decomp,allsub


def figures(overall,decomp,allsub):
    order=["subject_iid","prior_conditioned_iid","exchangeable_block_mixture","hmm"]
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4);w=.35
    for i,v in enumerate(["original","shuffled"]):
        g=overall[overall.data_version==v].set_index("model_name").loc[order];ax.bar(x+(i-.5)*w,g.ll_per_trial,w,label=v)
    ax.set(xticks=x,xticklabels=["Subject IID","Prior-conditioned IID","Block mixture","HMM"],ylabel="Held-out LL / trial",title="Same-fold held-out model comparison");ax.legend();fig.tight_layout();fig.savefig(FIG_DIR/"overall_model_comparison.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));g=decomp;xx=np.arange(len(g));ax.bar(xx-.18,g.hmm_original_gain,.36,label="HMM original - baseline");ax.bar(xx+.18,g.hmm_shuffled_gain,.36,label="HMM shuffled - baseline");ax.axhline(0,color="black",lw=1);ax.set(xticks=xx,xticklabels=g.baseline,ylabel="LL / trial gain",title="What remains after increasingly strong non-Markov baselines?");ax.legend();fig.tight_layout();fig.savefig(FIG_DIR/"markov_gain_decomposition.png",dpi=170);plt.close(fig)
    pivot=allsub.pivot_table(index="subject_id",columns=["model_name","data_version"],values="ll_per_trial")
    fig,ax=plt.subplots(figsize=(10,5));
    for name in order:ax.plot(pivot.index,pivot[(name,"original")],"o-",label=name)
    ax.set(xlabel="Subject",ylabel="Held-out LL / trial",title="Original-order performance for every subject",xticks=pivot.index);ax.legend(ncol=2);fig.tight_layout();fig.savefig(FIG_DIR/"subject_model_comparison.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));delta=pivot[("hmm","original")]-pivot[("exchangeable_block_mixture","original")];ax.bar(delta.index,delta);ax.axhline(0,color="black",lw=1);ax.set(xlabel="Subject",ylabel="HMM - block mixture LL / trial",title="Remaining original-order gain beyond exchangeable block heterogeneity",xticks=delta.index);fig.tight_layout();fig.savefig(FIG_DIR/"hmm_vs_block_mixture_by_subject.png",dpi=170);plt.close(fig)


def validate(original,shuffled,fold_df,overall):
    inv=pd.read_csv(COMP_DIR/"iid_order_invariance.csv")
    checks={"valid_rows_each":len(original)==len(shuffled)==83210,"same_trial_set":set(original.original_row_index)==set(shuffled.original_row_index),
            "same_folds":original.set_index("original_row_index").fold_id.sort_index().equals(shuffled.set_index("original_row_index").fold_id.sort_index()),
            "no_block_fold_leakage":original.groupby(["subject_id","original_block_id"]).fold_id.nunique().max()==1,
            "288_fold_rows":len(fold_df)==288,"all_iid_models_order_invariant":bool(inv.order_invariant_pass.all()),
            "finite_likelihoods":bool(np.isfinite(fold_df[["train_final_log_likelihood","test_total_log_likelihood","test_ll_per_trial"]]).all().all()),
            "all_folds_completed":bool((fold_df.groupby(["model_name","data_version","subject_id"]).fold_id.nunique()==4).all()),
            "comparison_has_hmm":set(overall.model_name)==set(MODEL_NAMES+["hmm"])}
    (LOG_DIR/"validation.json").write_text(json.dumps({k:bool(v) for k,v in checks.items()},indent=2),encoding="utf-8")
    if not all(checks.values()):raise AssertionError(checks)
    return checks


def write_report(overall,decomp,allsub,fold_df):
    wide=overall.pivot(index="model_name",columns="data_version",values="ll_per_trial")
    subj=allsub.pivot_table(index="subject_id",columns=["model_name","data_version"],values="ll_per_trial")
    block_delta=subj[("hmm","original")]-subj[("exchangeable_block_mixture","original")]
    hmm_order=wide.loc["hmm","original"]-wide.loc["hmm","shuffled"]
    text=f"""# IID and non-Markov baseline report

## Final conclusion

The exchangeable block mixture is the best same-fold held-out model ({wide.loc['exchangeable_block_mixture','original']:.6f} LL/trial). It has no trial-to-trial transitions yet outperforms the original HMM ({wide.loc['hmm','original']:.6f}) and shuffled HMM ({wide.loc['hmm','shuffled']:.6f}). The original HMM exceeds the block mixture for only {(block_delta>0).sum()}/12 subjects. This supports block-level strategy heterogeneity as the main source of the apparently sticky structure. Original order still contributes {hmm_order:+.6f} LL/trial within the HMM, but a strong first-order Markov transition is not required for the best prediction.

## Question and test sequence

All models use the same 83,210 valid trials, subject-specific four-fold run/block assignments, circular sensory/prior/lapse emissions, and training-only parameter fitting. The sequence was:

1. Subject IID mixture: fixed subject-level sensory/prior/lapse weights; no history.
2. Prior-conditioned IID mixture: weights depend only on the currently known `prior_std`; no history.
3. Exchangeable block mixture: a block has one latent type and trials are IID conditional on that type. Past responses update the block-type posterior causally, but there is no trial-to-trial state transition and the block likelihood is order-invariant.
4. Original three-state HMM: previous filtered state predicts the next state through a transition matrix.

Original and shuffled IID/non-Markov fits are numerically identical by construction; this was verified fold by fold.

## Weighted held-out results

| Model | Original LL/trial | Shuffled LL/trial |
|---|---:|---:|
| Subject IID | {wide.loc['subject_iid','original']:.6f} | {wide.loc['subject_iid','shuffled']:.6f} |
| Prior-conditioned IID | {wide.loc['prior_conditioned_iid','original']:.6f} | {wide.loc['prior_conditioned_iid','shuffled']:.6f} |
| Exchangeable block mixture | {wide.loc['exchangeable_block_mixture','original']:.6f} | {wide.loc['exchangeable_block_mixture','shuffled']:.6f} |
| HMM | {wide.loc['hmm','original']:.6f} | {wide.loc['hmm','shuffled']:.6f} |

## Decomposition

- Known prior condition gain over subject IID: {wide.loc['prior_conditioned_iid','original']-wide.loc['subject_iid','original']:+.6f} LL/trial.
- Exchangeable block heterogeneity gain over subject IID: {wide.loc['exchangeable_block_mixture','original']-wide.loc['subject_iid','original']:+.6f} LL/trial.
- HMM original gain over exchangeable block mixture: {wide.loc['hmm','original']-wide.loc['exchangeable_block_mixture','original']:+.6f} LL/trial.
- HMM shuffled gain over exchangeable block mixture: {wide.loc['hmm','shuffled']-wide.loc['exchangeable_block_mixture','shuffled']:+.6f} LL/trial.
- Original-order HMM advantage: {hmm_order:+.6f} LL/trial; subjects with HMM original above block mixture: {(block_delta>0).sum()}/12.

## Numerical status

- Baseline fold-result rows completed: {len(fold_df)}/288. Original/shuffled rows share the same fitted parameters; unique training fits converged: {int(fold_df.query("data_version == 'original'").train_converged.sum())}/144.
- Prior-conditioned IID, exchangeable block mixture, and HMM each have 15 subject-level free parameters (seven emission kappas plus eight mixture/sequence parameters), so the block-mixture advantage is not due to a larger parameter count than the HMM.
- No train/test block overlap, missing or duplicated trials, non-finite likelihoods, or order-invariance failures.
- Existing HMM and shuffle outputs were read only and protected by before/after SHA-256 checks.

## Interpretation

The subject IID comparison measures the total benefit of any adaptive latent-state model beyond single-trial emissions. The prior-conditioned IID removes effects explained by known prior width. The exchangeable block mixture removes predictable block-level strategy composition without introducing trial-to-trial transitions. Therefore, the HMM advantage beyond the block mixture is the most relevant estimate of additional within-block sequential information. The original-minus-shuffled HMM difference remains the clean order-specific control.
"""
    (ROOT/"IID_final_report.md").write_text(text,encoding="utf-8")


def manifest():
    rows=[]
    for p in sorted(x for x in ROOT.rglob("*") if x.is_file() and x.name!="output_manifest.csv" and "__pycache__" not in x.parts):
        rows.append({"relative_path":p.relative_to(ROOT).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256(p)})
    pd.DataFrame(rows).to_csv(LOG_DIR/"output_manifest.csv",index=False)


def main(force=True):
    ensure_dirs();before=protected_snapshot();(LOG_DIR/"protected_before.json").write_text(json.dumps(before,indent=2),encoding="utf-8")
    original,shuffled=load_data();base=base_module();full=run_full_fits(original,base);fold,preds=run_cv(original,shuffled,base)
    subject,overall,decomp,allsub=summaries(fold);figures(overall,decomp,allsub);checks=validate(original,shuffled,fold,overall);write_report(overall,decomp,allsub,fold)
    after=protected_snapshot();(LOG_DIR/"protected_after.json").write_text(json.dumps(after,indent=2),encoding="utf-8")
    if before!=after:raise AssertionError("Protected existing inputs changed")
    manifest();meta={"models":MODEL_NAMES,"valid_trials":len(original),"unique_training_fits":144,"fold_result_rows":len(fold),"validation":{k:bool(v) for k,v in checks.items()}}
    (LOG_DIR/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8");return overall,decomp,meta
