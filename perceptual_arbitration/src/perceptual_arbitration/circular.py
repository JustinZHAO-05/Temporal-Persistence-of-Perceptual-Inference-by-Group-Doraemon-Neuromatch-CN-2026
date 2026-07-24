from __future__ import annotations

import numpy as np
from scipy.special import i0e, logsumexp

TWO_PI = 2.0 * np.pi
LOG_TWO_PI = np.log(TWO_PI)
EPS = 1e-12


def deg2rad(x):
    return np.deg2rad(np.mod(x, 360.0))


def rad2deg(x):
    return np.mod(np.rad2deg(x), 360.0)


def circ_diff(a, b):
    """Signed circular difference a-b in radians, in [-pi, pi]."""
    return np.angle(np.exp(1j * (a - b)))


def circ_absdiff(a, b):
    return np.abs(circ_diff(a, b))


def wrap_rad(x):
    return np.mod(x, TWO_PI)


def log_i0(kappa):
    kappa = np.asarray(kappa)
    return np.log(i0e(kappa)) + np.abs(kappa)


def log_vonmises_from_delta(delta, kappa):
    return kappa * np.cos(delta) - LOG_TWO_PI - log_i0(kappa)


def log_vonmises(y, mu, kappa):
    return log_vonmises_from_delta(circ_diff(y, mu), kappa)


def kappa_from_R(R):
    """Approximate inverse of A(kappa)=I1(kappa)/I0(kappa).

    Based on common Best & Fisher approximations. Good enough for EM updates.
    """
    R = float(np.clip(R, 0.0, 0.999999999))
    if R < 1e-8:
        return 1e-8
    if R < 0.53:
        k = 2 * R + R**3 + 5 * R**5 / 6
    elif R < 0.85:
        k = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        denom = max(R**3 - 4 * R**2 + 3 * R, 1e-12)
        k = 1 / denom
    return float(np.clip(k, 1e-8, 5000.0))


def weighted_kappa(cos_delta, weights):
    weights = np.asarray(weights, dtype=float)
    denom = np.sum(weights)
    if denom <= EPS:
        return 1e-8
    R = np.sum(weights * cos_delta) / denom
    return kappa_from_R(R)


def stable_softmax(x, axis=-1):
    x = np.asarray(x)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def row_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def simplex_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    return x / max(float(x.sum()), eps)
