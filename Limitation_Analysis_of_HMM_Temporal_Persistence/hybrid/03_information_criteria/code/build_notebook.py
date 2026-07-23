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
    nbformat.v4.new_markdown_cell(
        "# 完整数据 AIC / BIC 审计\n\n"
        "比较原始三状态 HMM、exchangeable block mixture 和 block+Markov hybrid。"
    ),
    nbformat.v4.new_markdown_cell(
        "## 预先定义的计算口径\n\n"
        "每名被试独立拟合，因此逐被试计算 `AIC=-2LL+2k` 与 "
        "`BIC=-2LL+k log(n_subject)` 后求和。参数数目为 HMM=15、block mixture=15、"
        "block+Markov=33。另报告以总 trial 数计算的 pooled BIC，作为敏感性口径。"
    ),
    nbformat.v4.new_code_cell(
        "from pathlib import Path\n"
        "import sys, json, pandas as pd\n"
        "WORKSPACE = Path.cwd()\n"
        "TARGET = WORKSPACE/'Markov_test'/'09_information_criteria'\n"
        "for p in (WORKSPACE/'tmp'/'jupyter-notebook'/'runtime', TARGET, WORKSPACE/'Markov_test'/'model_artifacts'):\n"
        "    if str(p) not in sys.path: sys.path.insert(0, str(p))"
    ),
    nbformat.v4.new_code_cell(
        "from information_criteria_core import run_information_criteria\n"
        "subject_results, overall_results = run_information_criteria()\n"
        "overall_results"
    ),
    nbformat.v4.new_code_cell(
        "validation = json.loads((TARGET/'validation.json').read_text())\n"
        "display(pd.Series(validation, name='passed').to_frame())\n"
        "assert all(validation.values())\n"
        "display(subject_results.groupby('model_name')[['aic','bic_subject']].sum())"
    ),
    nbformat.v4.new_markdown_cell(
        "## 解释边界\n\n"
        "这些是完整训练数据的辅助拟合指标；主模型结论仍以 held-out one-step-ahead "
        "predictive likelihood 为准。对 mixture/HMM 的 BIC 近似需结合局部最优、标签对称性和收敛状态谨慎解释。"
    ),
]
nbformat.write(nb, out)
print(out)
