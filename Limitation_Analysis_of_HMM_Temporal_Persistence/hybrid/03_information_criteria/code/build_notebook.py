from pathlib import Path
import subprocess
import sys

import nbformat

folder = Path(__file__).resolve().parent
workspace = folder.parents[1]
helper = Path(r"C:\Users\baoba\.codex\skills\jupyter-notebook\scripts\new_notebook.py")
out = folder / "09_full_data_aic_bic.ipynb"

subprocess.run(
    [sys.executable, str(helper), "--kind", "experiment", "--title", "Full-data AIC/BIC audit", "--out", str(out)],
    check=True,
)
nb = nbformat.read(out, as_version=4)
nb.cells = [
    nbformat.v4.new_markdown_cell("# Full-data AIC/BIC audit\n\nCompare the original three-state HMM, exchangeable block mixture, and block-plus-Markov hybrid."),
    nbformat.v4.new_markdown_cell("## Predefined calculations\n\nCompute participant-wise AIC and BIC, then sum across participants. Parameter counts are HMM=15, block mixture=15, and hybrid=33. A pooled-trial BIC is also reported as a sensitivity analysis."),
    nbformat.v4.new_code_cell("from pathlib import Path\nimport sys, json, pandas as pd\nWORKSPACE = Path.cwd()\nTARGET = WORKSPACE/'Markov_test'/'09_information_criteria'\nfor p in (WORKSPACE/'tmp'/'jupyter-notebook'/'runtime', TARGET, WORKSPACE/'Markov_test'/'model_artifacts'):\n    if str(p) not in sys.path: sys.path.insert(0, str(p))"),
    nbformat.v4.new_code_cell("from information_criteria_core import run_information_criteria\nsubject_results, overall_results = run_information_criteria()\noverall_results"),
    nbformat.v4.new_code_cell("validation = json.loads((TARGET/'validation.json').read_text())\ndisplay(pd.Series(validation, name='passed').to_frame())\nassert all(validation.values())\ndisplay(subject_results.groupby('model_name')[['aic','bic_subject']].sum())"),
    nbformat.v4.new_markdown_cell("## Interpretation boundary\n\nThese are secondary full-training-data criteria. Primary conclusions rely on held-out one-step-ahead predictive likelihood. Interpret mixture/HMM BIC values together with local optima, label symmetry, and convergence status."),
]
nbformat.write(nb, out)
print(out)
