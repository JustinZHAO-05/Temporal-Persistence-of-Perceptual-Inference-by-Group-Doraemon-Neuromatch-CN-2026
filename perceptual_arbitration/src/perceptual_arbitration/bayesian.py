from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .circular import LOG_TWO_PI, TWO_PI, log_i0, circ_diff, deg2rad, wrap_rad, stable_softmax
from .data import DataBundle

GRID_DEG = np.arange(360.0)
GRID = deg2rad(GRID_DEG)


@dataclass
class BayesianParams:
    kappa_e: np.ndarray
    kappa_p: np.ndarray
    kappa_m: float
    lapse: float


@dataclass
class SwitchingParams:
    kappa_e: np.ndarray
    kappa_p: np.ndarray
    kappa_m: float
    lapse: float


@dataclass
class OptimFit:
    params: object
    train_loglik: float
    converged: bool
    seed: int
    message: str


def _vm_grid(center: float, kappa: float) -> np.ndarray:
    logp = kappa * np.cos(circ_diff(GRID, center)) - LOG_TWO_PI - log_i0(kappa)
    p = np.exp(logp)
    return p / p.sum()


def _nearest_grid_probs(y: np.ndarray, dist_grid: np.ndarray) -> np.ndarray:
    deg = np.mod(np.rad2deg(y), 360.0)
    lo = np.floor(deg).astype(int) % 360
    hi = (lo + 1) % 360
    frac = deg - np.floor(deg)
    return (1 - frac) * dist_grid[lo] + frac * dist_grid[hi]


def _pack_initial(data: DataBundle, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    base_e = np.interp(np.arange(ncoh), [0, max(ncoh - 1, 1)], [1.5, 18.0])
    base_p = np.maximum(0.5, 30.0 * (np.min(data.prior_values) / data.prior_values))
    log_ke = np.log(base_e * rng.lognormal(0, 0.25, ncoh))
    log_kp = np.log(base_p * rng.lognormal(0, 0.25, nprior))
    log_km = np.log(25.0 * rng.lognormal(0, 0.2))
    lapse_logit = np.log(0.02 / 0.98) + rng.normal(0, 0.2)
    return np.r_[log_ke, log_kp, log_km, lapse_logit]


def _unpack_bayes(v: np.ndarray, data: DataBundle) -> BayesianParams:
    ncoh = len(data.coh_values)
    nprior = len(data.prior_values)
    pos = 0
    kappa_e = np.exp(v[pos:pos + ncoh]); pos += ncoh
    kappa_p = np.exp(v[pos:pos + nprior]); pos += nprior
    kappa_m = float(np.exp(v[pos])); pos += 1
    lapse = float(1.0 / (1.0 + np.exp(-v[pos])))
    return BayesianParams(kappa_e=kappa_e, kappa_p=kappa_p, kappa_m=kappa_m, lapse=lapse)


def _unpack_switch(v: np.ndarray, data: DataBundle) -> SwitchingParams:
    p = _unpack_bayes(v, data)
    return SwitchingParams(kappa_e=p.kappa_e, kappa_p=p.kappa_p, kappa_m=p.kappa_m, lapse=p.lapse)


def _condition_keys(data: DataBundle, idx: np.ndarray):
    # Group by true direction, coherence index, prior index, prior mean.
    df = data.df.iloc[idx]
    for key, g in df.groupby(["motion_direction", "motion_coherence", "prior_std", "prior_mean"], sort=False):
        yield key, g.index.to_numpy(dtype=int)


def bayesian_condition_distribution(theta_deg: float, coh_value: float, prior_std: float, prior_mean_deg: float, params: BayesianParams, data: DataBundle, readout: Literal["map", "mean", "sample"] = "map") -> np.ndarray:
    ci = int(np.where(data.coh_values == coh_value)[0][0])
    pi = int(np.where(data.prior_values == prior_std)[0][0])
    ke = float(params.kappa_e[ci])
    kp = float(params.kappa_p[pi])
    theta = deg2rad(theta_deg)
    mu = deg2rad(prior_mean_deg)
    prior = _vm_grid(mu, kp)
    evidence_prob = _vm_grid(theta, ke)  # p(theta_e | theta_true) over evidence centers.
    response_dist = np.zeros(360)
    motor_cache = {}

    for e_idx, pe in enumerate(evidence_prob):
        if pe < 1e-12:
            continue
        likelihood = _vm_grid(GRID[e_idx], ke)
        posterior = likelihood * prior
        posterior = posterior / posterior.sum()
        if readout == "map":
            percept_idx = int(np.argmax(posterior))
            motor = motor_cache.get(percept_idx)
            if motor is None:
                motor = _vm_grid(GRID[percept_idx], params.kappa_m)
                motor_cache[percept_idx] = motor
            response_dist += pe * motor
        elif readout == "mean":
            z = np.sum(posterior * np.exp(1j * GRID))
            mean_angle = np.angle(z) % TWO_PI
            response_dist += pe * _vm_grid(mean_angle, params.kappa_m)
        else:  # sampling readout: response distribution is posterior convolved with motor.
            for percept_idx, pp in enumerate(posterior):
                if pp > 1e-8:
                    motor = motor_cache.get(percept_idx)
                    if motor is None:
                        motor = _vm_grid(GRID[percept_idx], params.kappa_m)
                        motor_cache[percept_idx] = motor
                    response_dist += pe * pp * motor
    response_dist = (1.0 - params.lapse) * response_dist + params.lapse * (np.ones(360) / 360.0)
    return response_dist / response_dist.sum()


def switching_condition_distribution(theta_deg: float, coh_value: float, prior_std: float, prior_mean_deg: float, params: SwitchingParams, data: DataBundle) -> np.ndarray:
    ci = int(np.where(data.coh_values == coh_value)[0][0])
    pi = int(np.where(data.prior_values == prior_std)[0][0])
    ke = float(params.kappa_e[ci])
    kp = float(params.kappa_p[pi])
    theta = deg2rad(theta_deg)
    mu = deg2rad(prior_mean_deg)
    p_prior = kp / max(kp + ke, 1e-12)
    p_sensory = 1.0 - p_prior
    sensory = _vm_grid(theta, ke)
    prior = _vm_grid(mu, params.kappa_m)  # prior mode reported with motor noise
    sensory_report = np.zeros(360)
    # sensory evidence center varies around true direction and motor noise adds another circular blur.
    evidence_prob = _vm_grid(theta, ke)
    motor_cache = {}
    for e_idx, pe in enumerate(evidence_prob):
        if pe < 1e-12:
            continue
        motor = motor_cache.get(e_idx)
        if motor is None:
            motor = _vm_grid(GRID[e_idx], params.kappa_m)
            motor_cache[e_idx] = motor
        sensory_report += pe * motor
    response = p_sensory * sensory_report + p_prior * prior
    response = (1.0 - params.lapse) * response + params.lapse * np.ones(360) / 360.0
    return response / response.sum()


def loglik_basic_bayesian(data: DataBundle, params: BayesianParams, seq_ids: Iterable[int], readout: Literal["map", "mean", "sample"] = "map") -> float:
    idx = np.concatenate([data.sequences[int(s)] for s in seq_ids])
    total = 0.0
    cache = {}
    for key, rows in _condition_keys(data, idx):
        if key not in cache:
            cache[key] = bayesian_condition_distribution(*key, params=params, data=data, readout=readout)
        probs = _nearest_grid_probs(data.y[rows], cache[key])
        total += np.log(np.maximum(probs, 1e-300)).sum()
    return float(total)


def loglik_condition_switching(data: DataBundle, params: SwitchingParams, seq_ids: Iterable[int]) -> float:
    idx = np.concatenate([data.sequences[int(s)] for s in seq_ids])
    total = 0.0
    cache = {}
    for key, rows in _condition_keys(data, idx):
        if key not in cache:
            cache[key] = switching_condition_distribution(*key, params=params, data=data)
        probs = _nearest_grid_probs(data.y[rows], cache[key])
        total += np.log(np.maximum(probs, 1e-300)).sum()
    return float(total)


def fit_basic_bayesian(data: DataBundle, seq_ids: Iterable[int], readout: Literal["map", "mean", "sample"] = "map", n_restarts: int = 10, seed0: int = 5000, maxiter: int = 500) -> OptimFit:
    best = None
    for r in range(n_restarts):
        x0 = _pack_initial(data, seed0 + r)
        def obj(v):
            p = _unpack_bayes(v, data)
            return -loglik_basic_bayesian(data, p, seq_ids, readout=readout)
        res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4})
        params = _unpack_bayes(res.x, data)
        ll = loglik_basic_bayesian(data, params, seq_ids, readout=readout)
        fit = OptimFit(params=params, train_loglik=ll, converged=bool(res.success), seed=seed0 + r, message=str(res.message))
        if best is None or fit.train_loglik > best.train_loglik:
            best = fit
    return best


def fit_condition_switching(data: DataBundle, seq_ids: Iterable[int], n_restarts: int = 10, seed0: int = 6000, maxiter: int = 500) -> OptimFit:
    best = None
    for r in range(n_restarts):
        x0 = _pack_initial(data, seed0 + r)
        def obj(v):
            p = _unpack_switch(v, data)
            return -loglik_condition_switching(data, p, seq_ids)
        res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4})
        params = _unpack_switch(res.x, data)
        ll = loglik_condition_switching(data, params, seq_ids)
        fit = OptimFit(params=params, train_loglik=ll, converged=bool(res.success), seed=seed0 + r, message=str(res.message))
        if best is None or fit.train_loglik > best.train_loglik:
            best = fit
    return best
