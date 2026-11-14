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
    inp = {"lamb_mhz": (1057.845, 1.0)}
    l = linear_sensitivity(hydrogen_crossing_gauss, inp)
    assert 0.5 < l["derivs"]["lamb_mhz"] < 0.6          # about 0.55 G per MHz of Lamb shift


def test_sodium_hyperfine_sampling_matches_exact_propagation():
    r = monte_carlo(sodium_3p12_splitting_mhz, {"a_3p12": (94.44, 0.13)}, n=500, seed=2)
    assert r["mean"] == pytest.approx(2 * 94.44, abs=0.05)
    assert r["std"] == pytest.approx(0.26, rel=0.15)    # splitting = 2A exactly, so sigma = 2 sigma_A


def test_monte_carlo_std_is_the_output_spread_and_sem_is_the_sampling_error_of_the_mean():
    """std does not shrink with n (it estimates the propagated spread); sem = std / sqrt(n) does."""
    f, inp = (lambda x: 2.0 * x), {"x": (1.0, 0.1)}
    small, large = monte_carlo(f, inp, 100, seed=5), monte_carlo(f, inp, 10000, seed=5)
    assert small["sem"] == pytest.approx(small["std"] / np.sqrt(100), rel=1e-12)
    assert large["sem"] == pytest.approx(large["std"] / np.sqrt(10000), rel=1e-12)
    assert small["std"] == pytest.approx(0.2, rel=0.2) and large["std"] == pytest.approx(0.2, rel=0.03)   # both estimate 2 sigma_x
    assert large["sem"] < small["sem"] / 5                                                                 # sem shrinks as 1/sqrt(n)
    assert large["sem"] < 0.02 * large["std"]                                                              # and is not an input uncertainty


def test_sampling_error_of_sigma_is_separate_from_sem():
    """Repeating the run with other seeds scatters std by ~1/sqrt(2(n-1)) of itself: a property of sampling, not physics."""
    stds = [monte_carlo(lambda x: x, {"x": (0.0, 1.0)}, 200, seed=s)["std"] for s in range(60)]
    assert np.std(stds, ddof=1) / np.mean(stds) == pytest.approx(1 / np.sqrt(2 * 199), rel=0.35)
