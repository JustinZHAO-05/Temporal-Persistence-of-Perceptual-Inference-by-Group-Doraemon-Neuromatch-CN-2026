# Within-block permutation control of the Markov interpretation

This experiment asks whether high HMM self-transition probabilities are
specific evidence for trial-to-trial Markov dependence. It compares the
original trial order with a reproducible within-segment permutation. Missing
responses create sequence boundaries before shuffling. The shuffled dataset
preserves each valid trial once, retains every row-level stimulus-response
pairing, and preserves block composition.

## Inputs and code

- Shuffled data: `data/data01_direction4priors_shuffled.csv`.
- Shared folds: `data/four_fold_assignments.csv`.
- Audit tables: `data/block_audit_table.csv`,
  `data/shuffle_block_report.csv`, and `data/missing_trial_report.csv`.
- Main implementation: `code/shuffle_control_core.py`.
- Executed notebooks: `code/HMM_shuffle_full_fit.ipynb` and
  `code/HMM_shuffle_4fold_heldout.ipynb`.

## Results

The held-out analysis completed 96 fits: 12 participants × 4 folds × 2 data
versions. The weighted LL/trial was -0.810573 for original order and -0.831003
for shuffled order, a difference of +0.020430 in favor of original order.
All 12 participant-level differences favored original order.

Only 66 of the 96 fold fits met the unchanged strict convergence criterion
(39/48 original and 27/48 shuffled). Non-converged fits were retained and
flagged. The contrast supports predictive information associated with order
under this shuffle design; it does not by itself identify the source or
mechanism of that information.

In the descriptive full-data fits, the mean diagonal transition probability
decreased from 0.6767 to 0.6233 after permutation but remained high relative to
the off-diagonal entries. The held-out decline shows that the original order
contains predictive information; it does not justify saying that order is
irrelevant. At the same time, the persistence of a diagonal-dominant transition
matrix after disrupting the original local order shows that a high diagonal is
not specific evidence for a first-order psychological state-switching process.
It may partly reflect longer-timescale participant or block-level stability.

The control therefore challenges a strong interpretation of the transition
matrix, not the descriptive usefulness of the three inferred HMM states.

See `reports/shuffle_control_report.md` for the original analysis report.
