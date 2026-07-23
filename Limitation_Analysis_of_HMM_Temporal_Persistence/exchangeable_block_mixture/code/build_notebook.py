from pathlib import Path
import subprocess,sys
helper=Path(r"C:\Users\baoba\.codex\skills\jupyter-notebook\scripts\new_notebook.py")
root=Path(__file__).resolve().parents[1];out=root/"IID_baseline_analysis.ipynb"
subprocess.run([sys.executable,str(helper),"--kind","experiment","--title","IID and non-Markov baselines for the three-state HMM","--out",str(out)],check=True)
import nbformat as n
nb=n.read(out,as_version=4)
nb.cells=[
 n.v4.new_markdown_cell("# IID and non-Markov baseline analysis\n\n**Objective.** Separate prediction due to single-trial emissions, known prior conditions, exchangeable block heterogeneity, and trial-to-trial HMM transitions using identical four-fold held-out trials."),
 n.v4.new_markdown_cell("## Pre-registered execution order\n\n1. Subject-level IID mixture.\n2. Prior-width-conditioned IID mixture.\n3. Three-class exchangeable block mixture with causal block-type updating but no state transitions.\n4. Compare all three with existing original/shuffled HMM results.\n\nThe three baselines are order-invariant; Original and Shuffled totals must match fold by fold."),
 n.v4.new_code_cell("from pathlib import Path\nimport sys,json,pandas as pd\nWORKSPACE=Path.cwd();RUNTIME=WORKSPACE/'tmp'/'jupyter-notebook'/'runtime';CORE=WORKSPACE/'IID'/'model_artifacts'\nfor p in (RUNTIME,CORE):\n    if str(p) not in sys.path:sys.path.insert(0,str(p))\nfrom iid_baseline_core import main\nprint({'workspace':str(WORKSPACE),'runtime':RUNTIME.exists()})"),
 n.v4.new_markdown_cell("## Execute full fits and all same-fold held-out tests\n\nEvery test response is scored with training-only parameters. No baseline uses future responses; the block mixture only updates its block-type posterior after each observed response."),
 n.v4.new_code_cell("overall,decomposition,metadata=main(force=True)\ndisplay(overall.sort_values(['data_version','ll_per_trial'],ascending=[True,False]))\ndisplay(decomposition)"),
 n.v4.new_markdown_cell("## Quality audit"),
 n.v4.new_code_cell("validation=json.loads((WORKSPACE/'IID'/'logs'/'validation.json').read_text())\ndisplay(pd.Series(validation,name='passed').to_frame())\nassert all(validation.values())"),
 n.v4.new_markdown_cell("## Interpretation boundary\n\nThe exchangeable block mixture is the strongest non-Markov baseline here: it can causally learn which kind of block it is observing, but its joint block likelihood is invariant to trial order. HMM improvement beyond this model is the relevant residual sequential gain; the original-minus-shuffled HMM contrast remains the direct order control.")]
n.write(nb,out);print(out)
