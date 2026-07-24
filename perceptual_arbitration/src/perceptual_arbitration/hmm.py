from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import logsumexp

from .circular import LOG_TWO_PI, log_i0, row_normalize, simplex_normalize
from .data import DataBundle

STATE_NAMES = ["S_sensory", "P_prior", "L_lapse"]
K = 3


@dataclass
class HMMParams:
    pi: np.ndarray
    A: np.ndarray
    kappa_s: np.ndarray
    kappa_p: np.ndarray

    def copy(self):
        return HMMParams(self.pi.copy(), self.A.copy(), self.kappa_s.copy(), self.kappa_p.copy())


@dataclass
class HMMFit:
    params: HMMParams
    train_loglik: float
    n_iter: int
    converged: bool
    seed: int
    history: list[float]
    restart_diagnostics: list[dict] | None = None


def init_hmm(data: DataBundle, seed: int = 0, sticky: bool = True) -> HMMParams:
    rng = np.random.default_rng(seed)
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)

    # Sensory precision should increase with coherence.
    base_s = np.interp(np.arange(ncoh), [0, max(ncoh - 1, 1)], [2.0, 18.0])
    # Prior precision should be larger for narrower priors. data.prior_values sorted ascending: 10,20,40,80.
    base_p = np.maximum(0.5, 30.0 * (np.min(data.prior_values) / data.prior_values))
    kappa_s = np.clip(base_s * rng.lognormal(0, 0.25, ncoh), 0.05, 500.0)
    kappa_p = np.clip(base_p * rng.lognormal(0, 0.25, nprior), 0.05, 500.0)

    if sticky:
        A = np.array([
            [0.85, 0.12, 0.03],
            [0.12, 0.85, 0.03],
            [0.45, 0.10, 0.45],
        ], dtype=float)
    else:
        A = np.ones((K, K), dtype=float) / K
    A += rng.gamma(shape=1.0, scale=0.01, size=(K, K))
    A = row_normalize(A)

    pi = np.array([0.65, 0.30, 0.05], dtype=float)
    pi += rng.gamma(shape=1.0, scale=0.01, size=K)
    pi = simplex_normalize(pi)
    return HMMParams(pi=pi, A=A, kappa_s=kappa_s, kappa_p=kappa_p)


def emission_logB(data: DataBundle, params: HMMParams, idx: np.ndarray) -> np.ndarray:
    """Log p(y_t|z_t) for rows idx, with state order S, P, L.

    Uses exponentially scaled Bessel normalization through ``log_i0`` so large
    concentration parameters do not overflow.
    """
    logB = np.empty((len(idx), K), dtype=float)
    ks = params.kappa_s[data.coh_idx[idx]]
    kp = params.kappa_p[data.prior_idx[idx]]
    logB[:, 0] = ks * data.cos_s[idx] - LOG_TWO_PI - log_i0(ks)
    logB[:, 1] = kp * data.cos_p[idx] - LOG_TWO_PI - log_i0(kp)
    logB[:, 2] = -LOG_TWO_PI
    return logB


emission_logB_stable = emission_logB


def forward_backward_log(logB: np.ndarray, logpi: np.ndarray, logA: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Forward-backward for a time-homogeneous HMM in log-space.

    Returns log-likelihood, gamma[T,K], xi[T-1,K,K].
    """
    T, K_ = logB.shape
    alpha = np.empty((T, K_), dtype=float)
    beta = np.empty((T, K_), dtype=float)
    alpha[0] = logpi + logB[0]
    for t in range(1, T):
        alpha[t] = logB[t] + logsumexp(alpha[t - 1][:, None] + logA, axis=0)
    ll = float(logsumexp(alpha[-1]))

    beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        beta[t] = logsumexp(logA + logB[t + 1][None, :] + beta[t + 1][None, :], axis=1)

    log_gamma = alpha + beta - ll
    gamma = np.exp(log_gamma)
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi = np.zeros((max(T - 1, 0), K_, K_), dtype=float)
    for t in range(T - 1):
        log_xi_t = alpha[t][:, None] + logA + logB[t + 1][None, :] + beta[t + 1][None, :] - ll
        m = logsumexp(log_xi_t)
        xi[t] = np.exp(log_xi_t - m)
    return ll, gamma, xi


def forward_backward_scaled(logB: np.ndarray, pi: np.ndarray, A: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scaled forward-backward using log-emissions as input.

    The emission matrix is row-shifted before exponentiation, so this keeps the
    stable log-density emission path while avoiding repeated logsumexp calls in
    the Baum-Welch hot loop.
    """
    T, K_ = logB.shape
    row_shift = logB.max(axis=1)
    B = np.exp(logB - row_shift[:, None])
    alpha = np.empty((T, K_), dtype=float)
    beta = np.empty((T, K_), dtype=float)
    scale = np.empty(T, dtype=float)

    alpha[0] = pi * B[0]
    scale[0] = max(float(alpha[0].sum()), 1e-300)
    alpha[0] /= scale[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        scale[t] = max(float(alpha[t].sum()), 1e-300)
        alpha[t] /= scale[t]
    ll = float(np.log(scale).sum() + row_shift.sum())

    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[t + 1] * beta[t + 1])
        beta[t] /= scale[t + 1]

    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi = np.zeros((max(T - 1, 0), K_, K_), dtype=float)
    for t in range(T - 1):
        num = alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]
        den = float(num.sum())
        if den > 0.0 and np.isfinite(den):
            xi[t] = num / den
        else:
            xi[t] = A
            xi[t] /= np.maximum(xi[t].sum(), 1e-300)
    return ll, gamma, xi


def fit_hmm_em(
    data: DataBundle,
    seq_ids: Iterable[int],
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-7,
    min_iter: int = 10,
    dirichlet: float = 1e-3,
    verbose: bool = False,
) -> HMMFit:
    """Fully convergent Baum-Welch EM for the Hidden Markov Switching Observer."""
    params = init_hmm(data, seed=seed)
    seqs = [data.sequences[int(s)] for s in seq_ids]
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    history: list[float] = []
    converged = False

    for it in range(max_iter):
        total_ll = 0.0
        pi_sum = np.zeros(K)
        A_sum = np.zeros((K, K))
        coh_w = np.zeros(ncoh)
        coh_c = np.zeros(ncoh)
        prior_w = np.zeros(nprior)
        prior_c = np.zeros(nprior)

        for idx in seqs:
            logB = emission_logB(data, params, idx)
            ll, gamma, xi = forward_backward_scaled(logB, params.pi, params.A)
            total_ll += ll
            pi_sum += gamma[0]
            A_sum += xi.sum(axis=0)
            np.add.at(coh_w, data.coh_idx[idx], gamma[:, 0])
            np.add.at(coh_c, data.coh_idx[idx], gamma[:, 0] * data.cos_s[idx])
            np.add.at(prior_w, data.prior_idx[idx], gamma[:, 1])
            np.add.at(prior_c, data.prior_idx[idx], gamma[:, 1] * data.cos_p[idx])

        new_pi = simplex_normalize(pi_sum + dirichlet)
        new_A = row_normalize(A_sum + dirichlet)
        from .circular import kappa_from_R
        new_kappa_s = np.array([kappa_from_R(coh_c[g] / max(coh_w[g], 1e-300)) for g in range(ncoh)])
        new_kappa_p = np.array([kappa_from_R(prior_c[g] / max(prior_w[g], 1e-300)) for g in range(nprior)])
        params = HMMParams(new_pi, new_A, new_kappa_s, new_kappa_p)
        history.append(float(total_ll))

        if verbose and (it < 5 or it % 25 == 0):
            print(f"HMM seed={seed} iter={it:04d} ll={total_ll:.4f} diag={np.diag(new_A)}", flush=True)

        if it >= min_iter and len(history) >= 2:
            improvement = history[-1] - history[-2]
            rel = improvement / max(1.0, abs(history[-2]))
            if improvement >= -1e-5 and rel < tol:
                converged = True
                break

    return HMMFit(params=params, train_loglik=history[-1], n_iter=len(history), converged=converged, seed=seed, history=history)


def fit_hmm_multistart(
    data: DataBundle,
    seq_ids: Iterable[int],
    n_restarts: int = 25,
    seed0: int = 1000,
    max_iter: int = 1000,
    tol: float = 1e-7,
    verbose: bool = False,
    n_jobs: int = 1,
) -> HMMFit:
    """Run many EM initializations and return the highest-likelihood converged solution."""
    from .multistart import run_multistart

    seq_ids = np.asarray(list(seq_ids), dtype=int)
    fit_data = data.fitting_view() if n_jobs != 1 else data

    def fit_seed(seed: int) -> HMMFit:
        return fit_hmm_em(fit_data, seq_ids, seed=seed, max_iter=max_iter, tol=tol, verbose=verbose)

    best, diagnostics = run_multistart(fit_seed, range(seed0, seed0 + n_restarts), n_jobs=n_jobs)
    best.restart_diagnostics = diagnostics
    return best


def loglik_hmm(data: DataBundle, params: HMMParams, seq_ids: Iterable[int]) -> float:
    total = 0.0
    for sid in seq_ids:
        idx = data.sequences[int(sid)]
        logB = emission_logB(data, params, idx)
        ll, _, _ = forward_backward_scaled(logB, params.pi, params.A)
        total += ll
    return float(total)


def posterior_hmm(data: DataBundle, params: HMMParams, seq_ids: Iterable[int]):
    import pandas as pd
    rows = []
    for sid in seq_ids:
        idx = data.sequences[int(sid)]
        logB = emission_logB(data, params, idx)
        _, gamma, _ = forward_backward_scaled(logB, params.pi, params.A)
        for k, row_idx in enumerate(idx):
            r = data.df.loc[row_idx]
            rows.append({
                "row_index": int(row_idx),
                "seq_id": int(sid),
                "subject_id": r["subject_id"],
                "session_id": r["session_id"],
                "run_id": r["run_id"],
                "trial_index": r["trial_index"],
                "estimate_deg": r["estimate_deg"],
                "motion_direction": r["motion_direction"],
                "motion_coherence": r["motion_coherence"],
                "prior_std": r["prior_std"],
                "prior_mean": r["prior_mean"],
                "p_sensory": gamma[k, 0],
                "p_prior": gamma[k, 1],
                "p_lapse": gamma[k, 2],
                "state_map": STATE_NAMES[int(np.argmax(gamma[k]))],
            })
    return pd.DataFrame(rows)
