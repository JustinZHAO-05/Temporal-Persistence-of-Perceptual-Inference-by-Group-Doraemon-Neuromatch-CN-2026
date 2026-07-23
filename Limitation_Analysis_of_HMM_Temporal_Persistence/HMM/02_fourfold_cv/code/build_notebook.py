from pathlib import Path

import nbformat


workspace = Path(__file__).resolve().parents[2]
notebook_path = workspace / "HMM_fourfold_comparison" / "notebooks" / "HMM_fourfold_heldout_comparison.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
notebook.cells = [
    nbformat.v4.new_markdown_cell(
        "# Four-fold held-out comparison for our revised HMM\n\n"
        "**Objective.** Refit the three-state subject-specific circular-emission soft-EM HMM in four run-level folds, "
        "score every held-out response with a response-before one-step-ahead density, and compare the result with "
        "the model benchmarks reported in the source HTML."
    ),
    nbformat.v4.new_markdown_cell(
        "## Evaluation contract\n\n"
        "- The split unit is the original subject-session-run; missing-response boundaries remain separate HMM segments.\n"
        "- Folds are deterministic and stratified by subject and prior width.\n"
        "- Every fold refits subject-specific parameters using training runs only.\n"
        "- Held-out segment trial 1 uses the training-derived subject-specific $\\pi$; later trials use filtering.\n"
        "- Smoothed posterior probabilities are never used for held-out prediction.\n"
        "- Existing result folders are read-protected by before/after SHA-256 snapshots."
    ),
    nbformat.v4.new_code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n\n"
        "WORKSPACE = Path.cwd()\n"
        "RUNTIME = WORKSPACE / 'tmp' / 'jupyter-notebook' / 'runtime'\n"
        "CORE_DIR = WORKSPACE / 'HMM_fourfold_comparison' / 'model_artifacts'\n"
        "for path in (RUNTIME, CORE_DIR):\n"
        "    if str(path) not in sys.path:\n"
        "        sys.path.insert(0, str(path))\n"
        "print({'workspace': str(WORKSPACE), 'runtime_exists': RUNTIME.exists(), 'core_exists': CORE_DIR.exists()})"
    ),
    nbformat.v4.new_markdown_cell(
        "## Run the complete experiment\n\n"
        "This cell performs all 48 subject-by-fold training fits, held-out scoring, bootstrap uncertainty, figures, "
        "the combined HTML report, manifests, and validation."
    ),
    nbformat.v4.new_code_cell(
        "from fourfold_cv_core import main\n\n"
        "run_metadata = main()\n"
        "run_metadata"
    ),
    nbformat.v4.new_markdown_cell("## Compact result audit"),
    nbformat.v4.new_code_cell(
        "import pandas as pd\n\n"
        "output_root = WORKSPACE / 'HMM_fourfold_comparison'\n"
        "overall = pd.read_csv(output_root / 'csv' / 'our_model_fourfold_overall_summary.csv')\n"
        "comparison = pd.read_csv(output_root / 'csv' / 'heldout_model_comparison.csv')\n"
        "validation = (output_root / 'logs' / 'FINAL_VALIDATION_REPORT.txt').read_text(encoding='utf-8')\n"
        "display(overall)\n"
        "display(comparison[['model', 'mean_held_out_ll_per_trial', 'ci_low', 'ci_high', 'value_provenance']])\n"
        "print(validation)"
    ),
    nbformat.v4.new_markdown_cell(
        "## Interpretation boundary\n\n"
        "The source HTML does not provide its original fold IDs or per-sequence scores. Therefore its displayed values "
        "are external benchmarks, not paired observations on these newly generated folds. A definitive paired bootstrap "
        "would require the original fold assignments or refitting all source models on the saved new folds."
    ),
]
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3 (HMM four-fold runtime)",
    "language": "python",
    "name": "python3",
}
notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
nbformat.write(notebook, notebook_path)
print(notebook_path)
