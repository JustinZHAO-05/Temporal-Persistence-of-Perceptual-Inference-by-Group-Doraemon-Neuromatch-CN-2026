from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .circular import LOG_TWO_PI, log_i0, kappa_from_R, simplex_normalize, stable_softmax
from .data import DataBundle
from .hmm import HMMParams, init_hmm, emission_logB_stable, K, STATE_NAMES


@dataclass
class CovariateHMMParams:
    pi: np.ndarray
    B: np.ndarray  # shape: previous-state K, next-state K, covariate P; row-wise multinomial logits.
    kappa_s: np.ndarray
    kappa_p: np.ndarray


@dataclass
class CovariateHMMFit:
    params: CovariateHMMParams
    train_loglik: float
    n_iter: int
    converged: bool
    seed: int
    history: list[float]
    restart_diagnostics: list[dict] | None = None


def init_covariate_hmm(data: DataBundle, seed: int = 0) -> CovariateHMMParams:
    base = init_hmm(data, seed=seed, sticky=True)
    rng = np.random.default_rng(seed)
    P = data.X_transition.shape[1]
    B = np.zeros((K, K, P), dtype=float)
    # Put initial homogeneous transition probabilities into intercept terms.
    B[:, :, 0] = np.log(base.A + 1e-12)
    B += rng.normal(0.0, 0.02, size=B.shape)
    return CovariateHMMParams(pi=base.pi, B=B, kappa_s=base.kappa_s, kappa_p=base.kappa_p)


def transition_logA_t(params: CovariateHMMParams, X_t: np.ndarray) -> np.ndarray:
    """Return log A_t[i,j] for one covariate vector X_t."""
    logits = np.einsum("ijp,p->ij", params.B, X_t)
    logits = logits - logsumexp(logits, axis=1, keepdims=True)
    return logits


def transition_logA_sequence(params: CovariateHMMParams, X: np.ndarray) -> np.ndarray:
    """Return logA[t,i,j] for transition into row t+1.

    If a sequence has T observations, X should have shape T,P. This returns T-1 matrices,
    where logA[t] uses covariates X[t+1] for transition z_t -> z_{t+1}.
    """
    if len(X) <= 1:
        return np.zeros((0, K, K), dtype=float)
    X_next = X[1:]
    logits = np.einsum("ijp,tp->tij", params.B, X_next)
    logits = logits - logsumexp(logits, axis=2, keepdims=True)
    return logits


def forward_backward_timevarying(logB: np.ndarray, logpi: np.ndarray, logA_seq: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    T, K_ = logB.shape
    alpha = np.empty((T, K_), dtype=float)
    beta = np.empty((T, K_), dtype=float)
    alpha[0] = logpi + logB[0]
    for t in range(1, T):
        alpha[t] = logB[t] + logsumexp(alpha[t - 1][:, None] + logA_seq[t - 1], axis=0)
    ll = float(logsumexp(alpha[-1]))
    beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        beta[t] = logsumexp(logA_seq[t] + logB[t + 1][None, :] + beta[t + 1][None, :], axis=1)
    log_gamma = alpha + beta - ll
    gamma = np.exp(log_gamma)
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    xi = np.zeros((max(T - 1, 0), K_, K_), dtype=float)
    for t in range(T - 1):
        lx = alpha[t][:, None] + logA_seq[t] + logB[t + 1][None, :] + beta[t + 1][None, :] - ll
        m = logsumexp(lx)
        xi[t] = np.exp(lx - m)
    return ll, gamma, xi


def _transition_objective_gradient(
    v: np.ndarray,
    X: np.ndarray,
    Xi_i: np.ndarray,
    l2: float,
    block_size: int = 2048,
) -> tuple[float, np.ndarray]:
    """Evaluate the transition objective without full-fold temporaries."""
    _, P = X.shape
    K_ = Xi_i.shape[1]
    B = np.zeros((K_, P), dtype=float)
    B[:-1] = v.reshape(K_ - 1, P)
    nll = 0.5 * l2 * np.sum(B[:-1] ** 2)
    grad = np.zeros_like(B)

    for start in range(0, len(X), block_size):
        stop = min(start + block_size, len(X))
        X_block = X[start:stop]
        Xi_block = Xi_i[start:stop]
        log_prob = X_block @ B.T
        log_prob -= logsumexp(log_prob, axis=1, keepdims=True)
        nll -= np.einsum("ij,ij->", Xi_block, log_prob)

        # Reuse the log-probability buffer for the residual so peak memory is
        # independent of the number of transitions in the fold.
        np.exp(log_prob, out=log_prob)
        log_prob *= Xi_block.sum(axis=1, keepdims=True)
        log_prob -= Xi_block
        grad += log_prob.T @ X_block

    grad[:-1] += l2 * B[:-1]
    return float(nll), grad[:-1].reshape(-1)


def _fit_transition_row(X: np.ndarray, Xi_i: np.ndarray, beta0: np.ndarray, l2: float) -> np.ndarray:
    """Weighted multinomial logistic M-step for one previous-state row.

    X: N transitions x P
    Xi_i: N transitions x K expected counts for i->j.
    beta0: K x P initial coefficients.
    To identify the multinomial model, last class coefficients are fixed to zero.
    """
    N, P = X.shape
    if N == 0 or Xi_i.sum() < 1e-12:
        return beta0
    K_ = Xi_i.shape[1]
    x0 = beta0[:-1].reshape(-1)

    def unpack(v):
        B = np.zeros((K_, P), dtype=float)
        B[:-1] = v.reshape(K_ - 1, P)
        return B

    res = minimize(
        lambda v: _transition_objective_gradient(v, X, Xi_i, l2),
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-8},
    )
    if not res.success:
        # Keep best iterate anyway; L-BFGS-B often stops with benign precision messages.
        pass
    B_new = unpack(res.x)
    # Add back an arbitrary row-centering so coefficients are easier to inspect.
    B_new = B_new - B_new.mean(axis=0, keepdims=True)
    return B_new


def fit_covariate_hmm_em(
    data: DataBundle,
    seq_ids: Iterable[int],
    seed: int = 0,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_iter: int = 10,
    l2_transition: float = 1.0,
    dirichlet_pi: float = 1e-3,
    verbose: bool = False,
) -> CovariateHMMFit:
    params = init_covariate_hmm(data, seed=seed)
    seqs = [data.sequences[int(s)] for s in seq_ids]
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    history: list[float] = []
    converged = False
    n_transitions = sum(max(len(idx) - 1, 0) for idx in seqs)
    n_covariates = data.X_transition.shape[1]

    with TemporaryDirectory(prefix=f"perceptual_covariate_{seed}_") as temp_name:
        if n_transitions:
            temp_path = Path(temp_name)
            X_all = np.memmap(
                temp_path / "transition_covariates.dat",
                dtype=np.float64,
                mode="w+",
                shape=(n_transitions, n_covariates),
            )
            Xi = np.memmap(
                temp_path / "transition_counts.dat",
                dtype=np.float64,
                mode="w+",
                shape=(n_transitions, K, K),
            )
        else:
            X_all = np.empty((0, n_covariates), dtype=float)
            Xi = np.empty((0, K, K), dtype=float)

        offsets: list[tuple[int, int]] = []
        cursor = 0
        for idx in seqs:
            stop = cursor + max(len(idx) - 1, 0)
            if stop > cursor:
                X_all[cursor:stop] = data.X_transition[idx[1:]]
            offsets.append((cursor, stop))
            cursor = stop

        try:
            for it in range(max_iter):
                logpi = np.log(params.pi + 1e-300)
                total_ll = 0.0
                pi_sum = np.zeros(K)
                coh_w = np.zeros(ncoh)
                coh_c = np.zeros(ncoh)
                prior_w = np.zeros(nprior)
                prior_c = np.zeros(nprior)

                hmm_like = HMMParams(params.pi, np.ones((K, K)) / K, params.kappa_s, params.kappa_p)
                for idx, (start, stop) in zip(seqs, offsets):
                    logB = emission_logB_stable(data, hmm_like, idx)
                    logA_seq = transition_logA_sequence(params, data.X_transition[idx])
                    ll, gamma, xi = forward_backward_timevarying(logB, logpi, logA_seq)
                    total_ll += ll
                    pi_sum += gamma[0]
                    np.add.at(coh_w, data.coh_idx[idx], gamma[:, 0])
                    np.add.at(coh_c, data.coh_idx[idx], gamma[:, 0] * data.cos_s[idx])
                    np.add.at(prior_w, data.prior_idx[idx], gamma[:, 1])
                    np.add.at(prior_c, data.prior_idx[idx], gamma[:, 1] * data.cos_p[idx])
                    if stop > start:
                        Xi[start:stop] = xi

                new_pi = simplex_normalize(pi_sum + dirichlet_pi)
                new_kappa_s = np.array([
                    kappa_from_R(coh_c[g] / max(coh_w[g], 1e-300)) for g in range(ncoh)
                ])
                new_kappa_p = np.array([
                    kappa_from_R(prior_c[g] / max(prior_w[g], 1e-300)) for g in range(nprior)
                ])

                new_B = params.B.copy()
                for i in range(K):
                    new_B[i] = _fit_transition_row(X_all, Xi[:, i, :], params.B[i], l2=l2_transition)

                params = CovariateHMMParams(
                    pi=new_pi,
                    B=new_B,
                    kappa_s=new_kappa_s,
                    kappa_p=new_kappa_p,
                )
                history.append(float(total_ll))
                if verbose and (it < 5 or it % 10 == 0):
                    print(f"IOHMM seed={seed} iter={it:04d} ll={total_ll:.4f}", flush=True)
                if it >= min_iter and len(history) >= 2:
                    improvement = history[-1] - history[-2]
                    rel = improvement / max(1.0, abs(history[-2]))
                    if improvement >= -1e-5 and rel < tol:
                        converged = True
                        break

            result = CovariateHMMFit(
                params=params,
                train_loglik=history[-1],
                n_iter=len(history),
                converged=converged,
                seed=seed,
                history=history,
            )
        finally:
            for array in (X_all, Xi):
                if isinstance(array, np.memmap):
                    array.flush()
                    array._mmap.close()
        return result


def fit_covariate_hmm_multistart(data: DataBundle, seq_ids: Iterable[int], n_restarts: int = 10, seed0: int = 3000, max_iter: int = 500, tol: float = 1e-6, l2_transition: float = 1.0, verbose: bool = False, n_jobs: int = 1) -> CovariateHMMFit:
    from .multistart import run_multistart

    seq_ids = np.asarray(list(seq_ids), dtype=int)
    fit_data = data.fitting_view() if n_jobs != 1 else data

    def fit_seed(seed: int) -> CovariateHMMFit:
        return fit_covariate_hmm_em(
            fit_data,
            seq_ids,
            seed=seed,
            max_iter=max_iter,
            tol=tol,
            l2_transition=l2_transition,
            verbose=verbose,
        )

    best, diagnostics = run_multistart(fit_seed, range(seed0, seed0 + n_restarts), n_jobs=n_jobs)
    best.restart_diagnostics = diagnostics
    return best


def loglik_covariate_hmm(data: DataBundle, params: CovariateHMMParams, seq_ids: Iterable[int]) -> float:
    total = 0.0
    logpi = np.log(params.pi + 1e-300)
    hmm_like = HMMParams(params.pi, np.ones((K, K)) / K, params.kappa_s, params.kappa_p)
    for sid in seq_ids:
        idx = data.sequences[int(sid)]
        logB = emission_logB_stable(data, hmm_like, idx)
        logA_seq = transition_logA_sequence(params, data.X_transition[idx])
        ll, _, _ = forward_backward_timevarying(logB, logpi, logA_seq)
        total += ll
    return float(total)


def average_transition_matrix(data: DataBundle, params: CovariateHMMParams, seq_ids: Iterable[int]) -> np.ndarray:
    mats = []
    for sid in seq_ids:
        idx = data.sequences[int(sid)]
        logA = transition_logA_sequence(params, data.X_transition[idx])
        if len(logA):
            mats.append(np.exp(logA))
    if not mats:
        return np.full((K, K), 1.0 / K)
    return np.concatenate(mats, axis=0).mean(axis=0)


def transition_posterior_statistics(
    data: DataBundle,
    params: CovariateHMMParams,
    seq_ids: Iterable[int],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return per-sequence transition covariates and expected transition counts."""
    logpi = np.log(params.pi + 1e-300)
    hmm_like = HMMParams(params.pi, np.ones((K, K)) / K, params.kappa_s, params.kappa_p)
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for sid_value in seq_ids:
        sid = int(sid_value)
        idx = data.sequences[sid]
        if len(idx) <= 1:
            continue
        logB = emission_logB_stable(data, hmm_like, idx)
        logA_seq = transition_logA_sequence(params, data.X_transition[idx])
        _, _, xi = forward_backward_timevarying(logB, logpi, logA_seq)
        rows.append((sid, data.X_transition[idx][1:].copy(), xi))
    return rows
