import numpy as np
from perceptual_arbitration.circular import circ_diff, deg2rad, rad2deg, kappa_from_R


def test_circ_diff_wraps():
    a = deg2rad(5)
    b = deg2rad(355)
    assert np.isclose(rad2deg(circ_diff(a, b)), 10.0)


def test_kappa_from_R_monotonic():
    assert kappa_from_R(0.2) < kappa_from_R(0.8)
