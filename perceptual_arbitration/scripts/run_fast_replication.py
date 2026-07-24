"""Fast replication of the preliminary static HMM vs independent switching result.

This is intentionally lighter than the publication run in scripts/run_all.py.
Use it first to check that the environment and CSV are working.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from perceptual_arbitration.data import load_direction_data, sequence_labels_for_cv
from perceptual_arbitration.hmm import fit_hmm_multistart, loglik_hmm
from perceptual_arbitration.independent_switching import fit_independent_multistart, loglik_independent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/fast")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = load_direction_data(args.csv)
    labels = sequence_labels_for_cv(data)
    seq_ids = np.arange(len(data.sequences))
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(seq_ids, labels), 1):
        train_seq = seq_ids[tr]; test_seq = seq_ids[te]
        ntest = int(data.seq_meta.loc[test_seq, "n"].sum())
        hmm = fit_hmm_multistart(data, train_seq, n_restarts=args.restarts, max_iter=args.max_iter, tol=args.tol, seed0=1000 + 100 * fold)
        ind = fit_independent_multistart(data, train_seq, n_restarts=args.restarts, max_iter=args.max_iter, tol=args.tol, seed0=2000 + 100 * fold)
        hmm_test = loglik_hmm(data, hmm.params, test_seq)
        ind_test = loglik_independent(data, ind.params, test_seq)
        rows.append({"fold": fold, "hmm_test_ll": hmm_test, "ind_test_ll": ind_test, "delta": hmm_test - ind_test, "delta_per_trial": (hmm_test - ind_test) / ntest, "hmm_A_SS": hmm.params.A[0,0], "hmm_A_PP": hmm.params.A[1,1]})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "fast_cv.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
