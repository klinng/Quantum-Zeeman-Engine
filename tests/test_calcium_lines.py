"""Tests for src/calcium_lines.py (Zeeman line spectrum of the 4s^2 1S0 -> 4s4p manifold)."""
import numpy as np
import pytest

from src.calcium_lines import calcium_lines, dipole_state, term_strength
from src.two_electron import CA_LEVELS_CM1, fit_to_levels, singlet_admixture_in_3p1

Z, D, ET, _ = fit_to_levels(*(CA_LEVELS_CM1[k] for k in ("3P0", "3P1", "3P2", "1P1")))


def test_dipole_state_is_normalised_singlet():
    for q in (-1, 0, 1):
        assert np.linalg.norm(dipole_state(q)) == pytest.approx(1.0)


def test_sum_rule_each_polarisation():
    df = calcium_lines(0.8, Z, D, ET)
    for q in (-1, 0, 1):
        assert df[df.q == q].strength.sum() == pytest.approx(1.0)


def test_zero_field_only_j1_levels_radiate():
    assert term_strength(0.0, "3P0", Z, D, ET) < 1e-20
    assert term_strength(0.0, "3P2", Z, D, ET) < 1e-20


def test_3p1_strength_is_the_singlet_admixture():
    eps2 = singlet_admixture_in_3p1(Z, D, ET)
    assert term_strength(0.0, "3P1", Z, D, ET) == pytest.approx(3 * eps2, rel=1e-8)
    assert term_strength(0.0, "1P1", Z, D, ET) == pytest.approx(3 * (1 - eps2), rel=1e-8)


def test_3p0_strength_grows_as_b_squared():
    ratio = term_strength(0.02, "3P0", Z, D, ET) / term_strength(0.01, "3P0", Z, D, ET)
    assert ratio == pytest.approx(4.0, rel=1e-3)


def test_3p0_is_pi_polarised_only():
    df = calcium_lines(0.5, Z, D, ET)
    g = df[df.label == "3P0"]
    assert g[g.q != 0].strength.max() < 1e-20


def test_3p1_line_wavelength_is_657_nm():
    df = calcium_lines(0.0, Z, D, ET)
    g = df[(df.label == "3P1") & (df.strength > 1e-8)]
    assert g.wavelength_nm.mean() == pytest.approx(657.46, abs=0.01)


def test_3p1_sigma_components_split_by_g_mu_b_b():
    from src.constants import MU_B_EV, ev_to_wavenumber
    b = 1.0
    df = calcium_lines(b, Z, D, ET)
    g = df[(df.label == "3P1") & (df.strength > 1e-8)]
    sp, sm = g[g.q == 1].wavenumber_cm1.iloc[0], g[g.q == -1].wavenumber_cm1.iloc[0]
    from src.two_electron import g_factor
    expected = 2 * g_factor("3P1", Z, D, ET) * ev_to_wavenumber(MU_B_EV * b)
    assert abs(sp - sm) == pytest.approx(expected, rel=1e-3)
