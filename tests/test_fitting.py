"""Recover known parameters from synthetic 'measurements'."""
import numpy as np
import pytest

from src.fitting import fit_parameters
from src.solvers import energy_vs_field
from src.validation import zeta_from_fine_structure

ZETA_TRUE = zeta_from_fine_structure(17.196, l=1)
B = np.linspace(0.5, 30, 12)


def test_recovers_zeta_exact_data():
    data = energy_vs_field(1, B, ZETA_TRUE)
    res = fit_parameters(1, B, data, zeta0_ev=0.5 * ZETA_TRUE)
    assert res.success
    assert res.zeta_ev == pytest.approx(ZETA_TRUE, rel=1e-6)


def test_recovers_zeta_noisy_data():
    rng = np.random.default_rng(0)
    data = energy_vs_field(1, B, ZETA_TRUE) + rng.normal(0, 1e-6, size=(B.size, 6))   # 1 micro-eV noise
    res = fit_parameters(1, B, data, zeta0_ev=2 * ZETA_TRUE)
    assert res.zeta_ev == pytest.approx(ZETA_TRUE, rel=0.01)


def test_recovers_zeta_and_g_s():
    data = energy_vs_field(1, B, ZETA_TRUE, g_s=2.05)
    res = fit_parameters(1, B, data, zeta0_ev=ZETA_TRUE, g_s0=2.0, fit_g_s=True)
    assert res.zeta_ev == pytest.approx(ZETA_TRUE, rel=1e-5)
    assert res.g_s == pytest.approx(2.05, rel=1e-5)
