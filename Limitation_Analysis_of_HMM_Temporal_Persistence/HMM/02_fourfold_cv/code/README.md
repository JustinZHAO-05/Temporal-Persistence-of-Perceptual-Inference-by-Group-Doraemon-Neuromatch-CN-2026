# Code inventory

- `fourfold_cv_core.py` is the primary reusable implementation and validated
  command-line entry point.
- `HMM_fourfold_heldout_comparison.ipynb` is the executed analysis notebook.
- `post_run_validation.py` contains the original post-run checks.
- `build_notebook.py` and `execute_notebook.py` are original notebook-generation
  helpers retained for provenance.
- The HTML inspection and injection scripts reproduce the original comparison
  report workflow and are not required for fitting the HMM.

Run `fourfold_cv_core.py` from the parent experiment directory. If complete
trial and fit tables already exist, the program performs a validation-only
rerun without refitting the 48 folds.
