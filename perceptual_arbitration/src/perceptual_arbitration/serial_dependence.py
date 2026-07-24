from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .circular import LOG_TWO_PI, log_i0, circ_diff, wrap_rad, stable_softmax
from .data import DataBundle


@dataclass
class SerialParams:
    weights: np.ndarray
    kappa_s: np.ndarray
    kappa_p: np.ndarray
    alpha_stim: float
    alpha_resp: float


@dataclass
class SerialFit:
    params: SerialParams
    train_loglik: float
    converged: bool
    seed: int
    result_message: str
    n_iter: int
    restart_diagnostics: list[dict] | None = None


def previous_arrays(data: DataBundle) -> tuple[np.ndarray, np.ndarray]:
    prev_theta = data.theta.copy()
    prev_y = data.y.copy()
    for idx in data.sequences:
        prev_theta[idx[0]] = data.theta[idx[0]]
        prev_y[idx[0]] = data.y[idx[0]]
        prev_theta[idx[1:]] = data.theta[idx[:-1]]
        prev_y[idx[1:]] = data.y[idx[:-1]]
    return prev_theta, prev_y


def sensory_center_with_serial(data: DataBundle, alpha_stim: float, alpha_resp: float) -> np.ndarray:
    prev_theta, prev_y = previous_arrays(data)
    shift = alpha_stim * circ_diff(prev_theta, data.theta) + alpha_resp * circ_diff(prev_y, data.theta)
    return wrap_rad(data.theta + shift)


def unpack(v: np.ndarray, ncoh: int, nprior: int, mode: Literal["stim", "resp", "both"] = "both") -> SerialParams:
    pos = 0
    weights = stable_softmax(v[pos:pos + 3]); pos += 3
    kappa_s = np.exp(v[pos:pos + ncoh]); pos += ncoh
    kappa_p = np.exp(v[pos:pos + nprior]); pos += nprior
    if mode == "stim":
        alpha_stim = 0.75 * np.tanh(v[pos]); pos += 1
        alpha_resp = 0.0
    elif mode == "resp":
        alpha_stim = 0.0
        alpha_resp = 0.75 * np.tanh(v[pos]); pos += 1
    else:
        alpha_stim = 0.75 * np.tanh(v[pos]); pos += 1
        alpha_resp = 0.75 * np.tanh(v[pos]); pos += 1
    return SerialParams(weights=weights, kappa_s=kappa_s, kappa_p=kappa_p, alpha_stim=alpha_stim, alpha_resp=alpha_resp)


def pack_initial(data: DataBundle, seed: int = 0, mode: str = "both") -> np.ndarray:
    rng = np.random.default_rng(seed)
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    logits_w = np.log(np.array([0.65, 0.30, 0.05])) + rng.normal(0, 0.05, 3)
    base_s = np.interp(np.arange(ncoh), [0, max(ncoh - 1, 1)], [2.0, 18.0])
    base_p = np.maximum(0.5, 30.0 * (np.min(data.prior_values) / data.prior_values))
    v = [*logits_w, *np.log(base_s * rng.lognormal(0, 0.2, ncoh)), *np.log(base_p * rng.lognormal(0, 0.2, nprior))]
    if mode in ("stim", "resp"):
        v += [rng.normal(0, 0.1)]
    else:
        v += [rng.normal(0, 0.1), rng.normal(0, 0.1)]
    return np.array(v, dtype=float)


def loglik_serial(data: DataBundle, params: SerialParams, seq_ids: Iterable[int]) -> float:
    idx = np.concatenate([data.sequences[int(s)] for s in seq_ids])
    center_s = sensory_center_with_serial(data, params.alpha_stim, params.alpha_resp)
    cos_s = np.cos(circ_diff(data.y[idx], center_s[idx]))
    cos_p = data.cos_p[idx]
    ks = params.kappa_s[data.coh_idx[idx]]
    kp = params.kappa_p[data.prior_idx[idx]]
    logB = np.empty((len(idx), 3), dtype=float)
    logB[:, 0] = ks * cos_s - LOG_TWO_PI - log_i0(ks)
    logB[:, 1] = kp * cos_p - LOG_TWO_PI - log_i0(kp)
    logB[:, 2] = -LOG_TWO_PI
    return float(np.sum(logsumexp(np.log(params.weights + 1e-300)[None, :] + logB, axis=1)))


def fit_serial_baseline(
    data: DataBundle,
    seq_ids: Iterable[int],
    mode: Literal["stim", "resp", "both"] = "both",
    n_restarts: int = 25,
    seed0: int = 4000,
    maxiter: int = 1000,
    l2_alpha: float = 0.0,
    n_jobs: int = 1,
) -> SerialFit:
    from .multistart import run_multistart

    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    seq_ids = np.asarray(list(seq_ids), dtype=int)
    fit_data = data.fitting_view() if n_jobs != 1 else data

    def objective(v):
        p = unpack(v, ncoh, nprior, mode=mode)
        ll = loglik_serial(fit_data, p, seq_ids)
        penalty = l2_alpha * (p.alpha_stim ** 2 + p.alpha_resp ** 2)
        return -ll + penalty

    def fit_seed(seed: int) -> SerialFit:
        x0 = pack_initial(fit_data, seed=seed, mode=mode)
        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-8, "maxls": 50})
        params = unpack(res.x, ncoh, nprior, mode=mode)
        ll = loglik_serial(fit_data, params, seq_ids)
        return SerialFit(
            params=params,
            train_loglik=ll,
            converged=bool(res.success),
            seed=seed,
            result_message=str(res.message),
            n_iter=int(getattr(res, "nit", 0)),
        )

    best, diagnostics = run_multistart(fit_seed, range(seed0, seed0 + n_restarts), n_jobs=n_jobs)
    best.restart_diagnostics = diagnostics
    return best
