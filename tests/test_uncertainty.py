import numpy as np
import pytest

from src.uncertainty import (hydrogen_crossing_gauss, linear_sensitivity, monte_carlo, sigma_compare,
                             sodium_3p12_splitting_mhz)


def test_monte_carlo_linear_model_exact():
    r = monte_carlo(lambda x, y: 3 * x - 2 * y, {"x": (1.0, 0.1), "y": (2.0, 0.2)}, n=20000, seed=1)
    assert r["mean"] == pytest.approx(-1.0, abs=0.01)
    assert r["std"] == pytest.approx(np.hypot(0.3, 0.4), rel=0.03)


def test_linear_sensitivity_matches_algebra():
    l = linear_sensitivity(lambda x, y: x * y, {"x": (2.0, 0.1), "y": (3.0, 0.2)})
    assert l["derivs"]["x"] == pytest.approx(3.0, rel=1e-6)
    assert l["std"] == pytest.approx(np.hypot(3 * 0.1, 2 * 0.2), rel=1e-6)


def test_seed_is_reproducible():
    f = lambda x: x ** 2
    assert monte_carlo(f, {"x": (1, 0.1)}, 50, seed=4)["mean"] == monte_carlo(f, {"x": (1, 0.1)}, 50, seed=4)["mean"]


def test_sigma_compare_verdicts():
    assert sigma_compare(1.0, 0.1, 1.1, 0.1)["verdict"] == "consistent"
    assert sigma_compare(1.0, 0.1, 1.3, 0.1)["verdict"] == "borderline"
    assert sigma_compare(1.0, 0.1, 2.0, 0.1)["verdict"] == "disagreement"
    assert sigma_compare(1.0, 0.0, 1.0)["z"] == 0.0


def test_hydrogen_crossing_matches_main_code_and_slope():
    assert hydrogen_crossing_gauss() == pytest.approx(573.4, abs=0.05)
    inp = {"lamb_mhz": (1057.864, 1.0)}
    l = linear_sensitivity(hydrogen_crossing_gauss, inp)
    assert 0.5 < l["derivs"]["lamb_mhz"] < 0.6          # about 0.55 G per MHz of Lamb shift


def test_sodium_hyperfine_sampling_matches_exact_propagation():
    r = monte_carlo(sodium_3p12_splitting_mhz, {"a_3p12": (94.44, 0.13)}, n=500, seed=2)
    assert r["mean"] == pytest.approx(2 * 94.44, abs=0.05)
    assert r["std"] == pytest.approx(0.26, rel=0.15)    # splitting = 2A exactly, so sigma = 2 sigma_A
