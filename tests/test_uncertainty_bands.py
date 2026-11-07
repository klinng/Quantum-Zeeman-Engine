import numpy as np
import pytest

from src.uncertainty import curve_band
from src.uncertainty_bands import _h_s_energy, _na_3p12_levels


def test_curve_band_linear_model_matches_algebra():
    x = np.linspace(0, 2, 5)
    band = curve_band(lambda x, a, b: a * x + b, {"a": (1.0, 0.1), "b": (0.0, 0.2)}, x, n=20000, seed=1)
    assert band["mean"] == pytest.approx(x, abs=0.01)
    assert band["std"] == pytest.approx(np.sqrt((0.1 * x) ** 2 + 0.2 ** 2), rel=0.03)


def test_hydrogen_2s_band_equals_lamb_sigma():
    b = np.array([0.05, 0.057, 0.065])
    band = curve_band(_h_s_energy, {"lamb_mhz": (1057.845, 0.009)}, b, n=60, seed=2)
    assert band["std"] == pytest.approx(0.009, rel=0.3)               # the 2S energy moves one-to-one with the Lamb shift


def test_sodium_level_uncertainty_is_below_natural_width():
    band = curve_band(_na_3p12_levels, {"a_3p12": (94.44, 0.13)}, np.array([0.0, 0.05, 0.1]), n=60, seed=3)
    assert (band["std"] < 9.8).all()
    assert band["std"].max() == pytest.approx(0.26, abs=0.15)          # at most 2 sigma_A (F=2 vs F=1 spread)
