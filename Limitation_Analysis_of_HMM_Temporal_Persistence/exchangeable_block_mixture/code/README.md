# Code inventory

- `exchangeable_block_model.py` is the preferred model-specific public
  interface.
- `iid_baseline_core.py` is the original shared implementation of the subject
  IID, prior-conditioned IID, and exchangeable block-mixture baselines.
- `fourfold_cv_core.py` supplies the shared circular-emission and sequence
  utilities.
- `validate_exchangeable_block.py` performs a read-only participant-1 refit and
  compares its likelihood with the migrated full-fit result.
- `IID_baseline_analysis.ipynb` is the executed notebook.
- `build_notebook.py` and `execute_notebook.py` are original notebook helpers
  retained for provenance.

Public reuse should import `fit`, `score`, and `load_data` from
`exchangeable_block_model.py`.
