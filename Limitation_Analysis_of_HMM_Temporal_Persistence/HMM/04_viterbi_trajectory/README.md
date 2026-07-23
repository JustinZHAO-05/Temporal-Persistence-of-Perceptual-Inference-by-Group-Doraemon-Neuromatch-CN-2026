# Descriptive Viterbi trajectories

`code/build_viterbi_trajectories.py` reads the saved full-data trial posterior
files from `HMM/01_full_data_fit/results/`. It does not refit the HMM. A rerun
writes to `results/reproduced/` so the migrated original outputs remain intact.

## Outputs

- `results/all_subjects_viterbi_states.csv`: decoded state for every valid trial.
- `results/viterbi_state_summary_by_subject.csv`: participant-by-state counts.
- `results/all_subjects_viterbi_trajectories.svg`: trajectory overview.
- `results/manifest.csv`: source and dimensional audit.

The output contains 83,210 trials from 12 participants and 388 valid sequence
segments. Viterbi decoding conditions on the complete sequence within a block.
Apparent persistence in these paths is therefore descriptive and is not an
independent test of first-order Markov dependence.
