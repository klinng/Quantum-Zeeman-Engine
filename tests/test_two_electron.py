"""Tests for src/two_electron.py (s-p two-electron atom: singlet and triplet P terms)."""
import numpy as np
import pytest

from src.constants import ev_to_wavenumber as cm1, wavenumber_to_ev as ev
from src.two_electron import (CA_LEVELS_CM1, classify_states, fit_to_levels, g_factor, h_two_electron, operators,
                              singlet_admixture_in_3p1, zero_field_levels_cm1, intercombination_rate)

ZETA, DELTA = ev(105.0), ev(8400.0)       # round numbers, calcium-like


def test_operators_obey_angular_momentum_algebra():
    o = operators()
    assert np.allclose(o["Lx"] @ o["Ly"] - o["Ly"] @ o["Lx"], 1j * o["Lz"])
    assert np.allclose(o["Sx"] @ o["Sy"] - o["Sy"] @ o["Sx"], 1j * o["Sz"])


def test_hamiltonian_is_hermitian():
    h = h_two_electron(0.7, ZETA, DELTA, ev(15000))
    assert np.allclose(h, h.conj().T)


def test_degeneracies_at_zero_field():
    df = classify_states(0.0, ZETA, DELTA)
    assert df.label.value_counts().to_dict() == {"3P2": 5, "3P1": 3, "1P1": 3, "3P0": 1}


def test_lande_interval_rule_without_mixing():
    # with a huge singlet-triplet gap the singlet decouples and the triplet obeys (E2 - E1) : (E1 - E0) = 2 : 1
    lv = zero_field_levels_cm1(ev(100.0), ev(1e6))
    assert (lv["3P2"] - lv["3P1"]) / (lv["3P1"] - lv["3P0"]) == pytest.approx(2.0, rel=1e-3)
    assert lv["3P1"] - lv["3P0"] == pytest.approx(50.0, rel=1e-3)           # zeta / 2


def test_lande_g_factors_of_the_terms():
    # LS coupling: g(3P1) = g(3P2) = 3/2, g(1P1) = 1 (plus the g_s - 2 correction of 0.0023)
    for lab, ref in (("3P1", 1.5), ("3P2", 1.5), ("1P1", 1.0)):
        assert g_factor(lab, ZETA, DELTA) == pytest.approx(ref, abs=2e-3)
    with pytest.raises(ValueError):
        g_factor("3P0", ZETA, DELTA)


def test_singlet_admixture_matches_perturbation_theory():
    lv = zero_field_levels_cm1(ZETA, DELTA)
    eps2_pt = (105.0 / np.sqrt(2) / (lv["1P1"] - lv["3P1"])) ** 2
    assert singlet_admixture_in_3p1(ZETA, DELTA) == pytest.approx(eps2_pt, rel=0.02)


def test_admixture_vanishes_without_spin_orbit_coupling():
    assert singlet_admixture_in_3p1(ev(1e-6), DELTA) < 1e-12


def test_fit_recovers_known_parameters():
    z, d, et = ev(104.0), ev(8000.0), ev(15000.0)
    lv = zero_field_levels_cm1(z, d, et)
    z2, d2, et2, res = fit_to_levels(lv["3P0"], lv["3P1"], lv["3P2"], lv["1P1"])
    assert cm1(z2) == pytest.approx(104.0, rel=1e-5) and cm1(d2) == pytest.approx(8000.0, rel=1e-5)
    assert np.abs(res).max() < 1e-4


def test_calcium_levels_fit_consistently():
    # four numbers, three parameters: the leftover is a genuine check of the model (Lande interval rule for calcium).
    # This loose bound only guards against a broken fit; the fit does NOT reproduce the data to their precision (the 3P1
    # residual is 0.089 cm^-1). That known misfit is asserted explicitly in tests/test_calcium_audit.py (marker: spectroscopic).
    *_, res = fit_to_levels(*(CA_LEVELS_CM1[k] for k in ("3P0", "3P1", "3P2", "1P1")))
    assert np.abs(res).max() < 0.15


def test_calcium_3p1_energy_matches_the_measured_frequency():
    from src.two_electron import CA_3P1_MEASURED_HZ
    assert CA_LEVELS_CM1["3P1"] == pytest.approx(CA_3P1_MEASURED_HZ / 299792458.0 / 100.0, abs=1e-4)


def test_figure_builds():
    import matplotlib
    matplotlib.use("Agg")
    from src.two_electron_figures import fig_two_electron_levels
    assert fig_two_electron_levels(ZETA, DELTA, ev(15000.0), b_max=5.0, n=8) is not None


def test_intercombination_rate_scaling():
    # eps^2 times the singlet rate, scaled by the cube of the wavelength ratio
    assert intercombination_rate(1e-4, 2e8, 400.0, 800.0) == pytest.approx(1e-4 * 2e8 / 8)
