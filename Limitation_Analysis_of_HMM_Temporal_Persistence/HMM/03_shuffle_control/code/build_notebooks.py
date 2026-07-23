from pathlib import Path
import subprocess, sys

R=Path(__file__).resolve().parents[1]
helper=Path(r"C:\Users\baoba\.codex\skills\jupyter-notebook\scripts\new_notebook.py")

def scaffold(name,title):
    p=R/name
    subprocess.run([sys.executable,str(helper),"--kind","experiment","--title",title,"--out",str(p)],check=True)
    import nbformat as nbf
    nb=nbf.read(p,as_version=4)
    return p,nb,nbf

def code(nbf,s): return nbf.v4.new_code_cell(s)
def md(nbf,s): return nbf.v4.new_markdown_cell(s)

p,nb,n=scaffold("HMM_shuffle_full_fit.ipynb","Trial-order shuffle control: full shuffled fit")
nb.cells=[
 md(n,"# Trial-order shuffle control: full shuffled fit\n\n**Objective.** Audit sequence boundaries, create one deterministic row-wise shuffle within missing-delimited segments, and fit the final revised 3-state soft EM-HMM to all shuffled subjects."),
 md(n,"## Contract\n\n`p_*` are smoothed forward-backward state posteriors. `filtered_prob_*` use data through trial t. `prior_predictive_prob_*` and `one_step_predictive_*` are response-before predictions. Training likelihood is descriptive, not evidence of generalization."),
 code(n,"from pathlib import Path\nimport sys, json, pandas as pd\nWORKSPACE=Path.cwd()\nRUNTIME=WORKSPACE/'tmp'/'jupyter-notebook'/'runtime'\nCORE=WORKSPACE/'shuffle_control'/'model_artifacts'\nfor p in (RUNTIME,CORE):\n    if str(p) not in sys.path: sys.path.insert(0,str(p))\nfrom shuffle_control_core import protected_snapshot, prepare_data, run_full_fit\nbefore=protected_snapshot()\nprint({'workspace':str(WORKSPACE),'runtime':RUNTIME.exists()})"),
 md(n,"## Block audit and deterministic shuffled dataset"),
 code(n,"prepared=prepare_data(force=True)\ndisplay(pd.Series(prepared['audit'],name='value').to_frame())\ndisplay(prepared['block_report'].head())\ndisplay(prepared['missing'])"),
 md(n,"The model block is `subject_id + session_id + run_id`; all three conditioning fields are fixed within it. Missing-response trials are boundaries. The actual permutation unit is the resulting segment, using master seed 20260717 plus a stable sorted segment number."),
 md(n,"## Fit all shuffled subjects with the unchanged revised Soft EM-HMM"),
 code(n,"result=run_full_fit(force=False)\ndisplay(result['summary'])"),
 md(n,"## Full-fit validation"),
 code(n,"validation=json.loads((WORKSPACE/'shuffle_control'/'logs'/'full_fit_validation.json').read_text())\ndisplay(pd.Series(validation,name='passed').to_frame())\nassert all(validation.values())"),
 md(n,"## Interpretation boundary\n\nThe full-data likelihood and smoothed posterior summaries describe the fit. Predictive conclusions are deferred to the separate held-out notebook.")]
n.write(nb,p)

p,nb,n=scaffold("HMM_shuffle_4fold_heldout.ipynb","Trial-order shuffle control: four-fold held-out comparison")
nb.cells=[
 md(n,"# Trial-order shuffle control: four-fold held-out comparison\n\n**Objective.** Fit original and shuffled versions in the same four saved run-level folds and compare held-out one-step-ahead predictive density."),
 md(n,"## Evaluation contract\n\n- Original subject-session-run blocks retain identical fold membership across versions.\n- Every fold fits only the other three folds.\n- Test parameters are fixed; test data never update π, transitions, or kappas.\n- Each test segment restarts from the training-derived π.\n- `p_*` are smoothed state inference; `test_log_predictive_density` is the primary out-of-sample prediction."),
 code(n,"from pathlib import Path\nimport sys,json,pandas as pd\nWORKSPACE=Path.cwd(); RUNTIME=WORKSPACE/'tmp'/'jupyter-notebook'/'runtime'; CORE=WORKSPACE/'shuffle_control'/'model_artifacts'\nfor p in (RUNTIME,CORE):\n    if str(p) not in sys.path: sys.path.insert(0,str(p))\nfrom shuffle_control_core import protected_snapshot, run_heldout, finalize_protection\nbefore=protected_snapshot()"),
 md(n,"## Execute all 96 training fits and held-out scoring"),
 code(n,"result=run_heldout(force=True)\ndisplay(result['comparison'])"),
 md(n,"## Quality audit"),
 code(n,"validation=json.loads((WORKSPACE/'shuffle_control'/'logs'/'heldout_validation.json').read_text())\ndisplay(pd.Series(validation,name='passed').to_frame())\nassert all(validation.values())\nfinalize_protection(before)"),
 md(n,"## Primary interpretation"),
 code(n,"c=result['comparison']; display(pd.Series({'subjects_original_better':int((c.delta_test_ll_per_trial>0).sum()),'mean_subject_delta':c.delta_test_ll_per_trial.mean(),'weighted_original_ll':(result['folds'].query(\"data_version=='original'\").test_total_log_likelihood.sum()/result['folds'].query(\"data_version=='original'\").n_test_trials.sum()),'weighted_shuffled_ll':(result['folds'].query(\"data_version=='shuffled'\").test_total_log_likelihood.sum()/result['folds'].query(\"data_version=='shuffled'\").n_test_trials.sum())},name='value').to_frame())"),
 md(n,"The formal conclusion is recorded in `shuffle_control_report.md` and is based on held-out one-step-ahead likelihood, not on training likelihood or smoothed posteriors.")]
n.write(nb,p)
print('built',R)
