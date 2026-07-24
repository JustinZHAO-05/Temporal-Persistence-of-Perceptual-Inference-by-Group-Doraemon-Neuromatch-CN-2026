"""Optional slow baseline: Basic Bayesian MAP/sampling and condition-dependent Switching.

These grid-based models are computationally heavier than the HMM family. Run after
finishing the main HMM/serial-dependence comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from perceptual_arbitration.data import load_direction_data, sequence_labels_for_cv
from perceptual_arbitration.bayesian import fit_basic_bayesian, loglik_basic_bayesian, fit_condition_switching, loglik_condition_switching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/bayesian_baselines")
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = load_direction_data(args.csv)
    seq_ids = np.arange(len(data.sequences))
    labels = sequence_labels_for_cv(data)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(seq_ids, labels), 1):
        train_seq = seq_ids[tr]; test_seq = seq_ids[te]
        ntest = int(data.seq_meta.loc[test_seq, "n"].sum())
        sw = fit_condition_switching(data, train_seq, n_restarts=args.restarts, maxiter=args.maxiter)
        sw_test = loglik_condition_switching(data, sw.params, test_seq)
        rows.append({"fold": fold, "model": "Original_condition_switching", "test_ll": sw_test, "test_ll_per_trial": sw_test/ntest, "train_ll": sw.train_loglik, "converged": sw.converged})
        for readout in ["map", "mean", "sample"]:
            fit = fit_basic_bayesian(data, train_seq, readout=readout, n_restarts=args.restarts, maxiter=args.maxiter)
            test_ll = loglik_basic_bayesian(data, fit.params, test_seq, readout=readout)
            rows.append({"fold": fold, "model": f"Basic_Bayesian_{readout}", "test_ll": test_ll, "test_ll_per_trial": test_ll/ntest, "train_ll": fit.train_loglik, "converged": fit.converged})
        pd.DataFrame(rows).to_csv(out / "bayesian_cv_live.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "bayesian_cv.csv", index=False)


if __name__ == "__main__":
    main()
