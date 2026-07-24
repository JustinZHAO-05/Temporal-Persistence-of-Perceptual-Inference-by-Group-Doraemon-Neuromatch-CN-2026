from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import logsumexp

from .circular import LOG_TWO_PI, log_i0, simplex_normalize, kappa_from_R
from .data import DataBundle
from .hmm import STATE_NAMES, K


@dataclass
class IndepParams:
    w: np.ndarray
    kappa_s: np.ndarray
    kappa_p: np.ndarray


@dataclass
class IndepFit:
    params: IndepParams
    train_loglik: float
    n_iter: int
    converged: bool
    seed: int
    history: list[float]
    restart_diagnostics: list[dict] | None = None


def init_independent(data: DataBundle, seed: int = 0) -> IndepParams:
    rng = np.random.default_rng(seed)
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    base_s = np.interp(np.arange(ncoh), [0, max(ncoh - 1, 1)], [2.0, 18.0])
    base_p = np.maximum(0.5, 30.0 * (np.min(data.prior_values) / data.prior_values))
    w = np.array([0.65, 0.30, 0.05]) + rng.gamma(1.0, 0.01, K)
    w = simplex_normalize(w)
    return IndepParams(
        w=w,
        kappa_s=np.clip(base_s * rng.lognormal(0, 0.25, ncoh), 0.05, 500.0),
        kappa_p=np.clip(base_p * rng.lognormal(0, 0.25, nprior), 0.05, 500.0),
    )


def emission_logB(data: DataBundle, params: IndepParams, idx: np.ndarray) -> np.ndarray:
    logB = np.empty((len(idx), K), dtype=float)
    ks = params.kappa_s[data.coh_idx[idx]]
    kp = params.kappa_p[data.prior_idx[idx]]
    logB[:, 0] = ks * data.cos_s[idx] - LOG_TWO_PI - log_i0(ks)
    logB[:, 1] = kp * data.cos_p[idx] - LOG_TWO_PI - log_i0(kp)
    logB[:, 2] = -LOG_TWO_PI
    return logB


def fit_independent_em(
    data: DataBundle,
    seq_ids: Iterable[int],
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    min_iter: int = 10,
    dirichlet: float = 1e-3,
    verbose: bool = False,
) -> IndepFit:
    params = init_independent(data, seed=seed)
    idx = np.concatenate([data.sequences[int(s)] for s in seq_ids])
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    history: list[float] = []
    converged = False

    for it in range(max_iter):
        logB = emission_logB(data, params, idx)
        logpost = np.log(params.w + 1e-300)[None, :] + logB
        logden = logsumexp(logpost, axis=1)
        ll = float(np.sum(logden))
        gamma = np.exp(logpost - logden[:, None])

        w = simplex_normalize(gamma.sum(axis=0) + dirichlet)
        coh_w = np.zeros(ncoh)
        coh_c = np.zeros(ncoh)
        prior_w = np.zeros(nprior)
        prior_c = np.zeros(nprior)
        np.add.at(coh_w, data.coh_idx[idx], gamma[:, 0])
        np.add.at(coh_c, data.coh_idx[idx], gamma[:, 0] * data.cos_s[idx])
        np.add.at(prior_w, data.prior_idx[idx], gamma[:, 1])
        np.add.at(prior_c, data.prior_idx[idx], gamma[:, 1] * data.cos_p[idx])
        kappa_s = np.array([kappa_from_R(coh_c[g] / max(coh_w[g], 1e-300)) for g in range(ncoh)])
        kappa_p = np.array([kappa_from_R(prior_c[g] / max(prior_w[g], 1e-300)) for g in range(nprior)])
        params = IndepParams(w=w, kappa_s=kappa_s, kappa_p=kappa_p)
        history.append(ll)

        if verbose and (it < 5 or it % 25 == 0):
            print(f"IND seed={seed} iter={it:04d} ll={ll:.4f} w={w}", flush=True)
        if it >= min_iter and len(history) >= 2:
            improvement = history[-1] - history[-2]
            rel = improvement / max(1.0, abs(history[-2]))
            if improvement >= -1e-5 and rel < tol:
                converged = True
                break
    return IndepFit(params=params, train_loglik=history[-1], n_iter=len(history), converged=converged, seed=seed, history=history)


def fit_independent_multistart(data: DataBundle, seq_ids: Iterable[int], n_restarts: int = 25, seed0: int = 2000, max_iter: int = 1000, tol: float = 1e-8, verbose: bool = False, n_jobs: int = 1) -> IndepFit:
    from .multistart import run_multistart

    seq_ids = np.asarray(list(seq_ids), dtype=int)
    fit_data = data.fitting_view() if n_jobs != 1 else data

    def fit_seed(seed: int) -> IndepFit:
        return fit_independent_em(fit_data, seq_ids, seed=seed, max_iter=max_iter, tol=tol, verbose=verbose)

    best, diagnostics = run_multistart(fit_seed, range(seed0, seed0 + n_restarts), n_jobs=n_jobs)
    best.restart_diagnostics = diagnostics
    return best


def loglik_independent(data: DataBundle, params: IndepParams, seq_ids: Iterable[int]) -> float:
    idx = np.concatenate([data.sequences[int(s)] for s in seq_ids])
    logB = emission_logB(data, params, idx)
    logpost = np.log(params.w + 1e-300)[None, :] + logB
    return float(np.sum(logsumexp(logpost, axis=1)))
