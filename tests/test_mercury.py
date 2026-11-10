"""Tests for src/mercury.py (Hg 546.1 nm Zeeman pattern and the pressure-scanned Fabry-Perot analysis of the teaching lab)."""
import numpy as np
import pytest

from src.constants import MU_B_EV, ev_to_wavenumber
from src.mercury import (CM1_PER_TESLA, G_LOWER, G_UPPER, bohr_from_peak_spacing, bohr_uncertainty, components, dn_dp, index,
                         lande_g, predicted_peak_spacing_mmhg, pressure_scan)

CODATA_MU_B_HC_CM1_PER_GAUSS = 4.668644e-5          # mu_B / hc = 0.4668644 cm^-1 / T


def test_lande_values_of_the_two_levels():
    assert (G_UPPER, G_LOWER) == (2.0, 1.5)
    assert lande_g(1, 0, 1) == pytest.approx(1.0) and lande_g(0.5, 0.5, 0) == pytest.approx(2.0)


def test_lande_agrees_with_the_two_electron_engine():
    from src.two_electron import CA_LEVELS_CM1, fit_to_levels, g_factor
    z, d, et, _ = fit_to_levels(*(CA_LEVELS_CM1[k] for k in ("3P0", "3P1", "3P2", "1P1")))
    # the engine uses g_s = 2.0023, which adds (g_s - 2)/2 = 0.00116 to g(3P2); the handout's formula uses g_s = 2
    assert g_factor("3P2", z, d, et) == pytest.approx(G_LOWER + 0.5 * (2.00231930436 - 2), abs=1e-5)


def test_nine_components_delta_from_minus_2_to_2():
    df = components(0.234)
    assert len(df) == 9
    assert np.allclose(df.delta, np.arange(-2, 2.01, 0.5))


def test_strengths_obey_sum_rules_and_symmetry():
    df = components(0.234)
    for mu in (-1, 0, 1):
        assert df[df.m_upper == mu].strength.sum() == pytest.approx(5 / 3)       # same for every M_u
    assert np.allclose(df.strength.values, df.strength.values[::-1])                 # symmetric about delta = 0


def test_six_sigma_three_pi():
    df = components(0.234)
    assert (df.polarisation == "pi").sum() == 3 and (df.polarisation != "pi").sum() == 6


def test_zeeman_unit_equals_codata():
    assert CM1_PER_TESLA == pytest.approx(0.4668644, rel=1e-6)
    assert float(ev_to_wavenumber(MU_B_EV)) == pytest.approx(CM1_PER_TESLA)


def test_observed_pair_splitting_is_two_mu_b_b():
    df = components(0.234)
    a = df[(df.m_upper == 1) & (df.m_lower == 2)].dsigma_cm1.iloc[0]
    b = df[(df.m_upper == -1) & (df.m_lower == -2)].dsigma_cm1.iloc[0]
    assert abs(b - a) == pytest.approx(2 * CM1_PER_TESLA * 0.234)


def test_wavelength_shift_of_outer_lines():
    """Unit check against the handout's statement that the 435.8 nm line splits by about 0.0089 nm per tesla (delta = 1)."""
    shift_nm = (435.8e-7) ** 2 * CM1_PER_TESLA * 1e7
    assert shift_nm == pytest.approx(0.0089, abs=2e-4)


def test_index_law_and_slope():
    assert index(760, 273.0) == pytest.approx(1.0003)
    assert dn_dp(293.0) == pytest.approx(3.678e-7, rel=1e-3)


def test_predicted_peak_spacing_for_the_handouts_field():
    assert predicted_peak_spacing_mmhg(2340, 293.0) == pytest.approx(32.45, abs=0.05)


def test_handout_analysis_recovers_mu_b_from_simulated_peaks():
    """Simulate the recording (e = 10 mm assumed), pick the two strongest peaks of an order, apply the handout's formula."""
    p = np.linspace(100, 760, 400000)
    y = pressure_scan(p, 0.234, 10.0, 293.0)                       # all six sigma components, R = 0.9
    pk = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > 0.8))[0] + 1
    p1, p2 = p[pk[2]], p[pk[3]]                                      # a pair inside one order
    dp = p2 - p1
    assert dp == pytest.approx(predicted_peak_spacing_mmhg(2340, 293.0), abs=0.05)
    got = bohr_from_peak_spacing(dp, 2340, 293.0, p_mean_mmhg=(p1 + p2) / 2)
    assert got == pytest.approx(CODATA_MU_B_HC_CM1_PER_GAUSS, rel=2e-3)


def test_uncertainty_scales_with_reading_error():
    u1 = bohr_uncertainty(32.0, 1.0, 2340, 20, 293.0)
    u2 = bohr_uncertainty(32.0, 2.0, 2340, 20, 293.0)
    assert u2 > u1 and u1 / bohr_from_peak_spacing(32.0, 2340, 293.0) == pytest.approx(np.hypot(1 / 32, 20 / 2340))


def test_analyse_recording_recovers_spacing_and_mu_b():
    from src.mercury import analyse_recording
    p = np.linspace(100, 760, 400000)
    y = pressure_scan(p, 0.234, 10.0, 293.0)
    pk = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > 0.8))[0] + 1
    order = p[pk[2]] - p[pk[0]]                       # same component, successive orders
    pair = p[pk[3]] - p[pk[2]]
    r = analyse_recording(order, pair, 2340, 293.0, p_mean_mmhg=(p[pk[2]] + p[pk[3]]) / 2, sigma_dp_mmhg=1.0, sigma_b_gauss=20)
    assert r["e_mm"] == pytest.approx(10.0, rel=2e-3)
    assert abs(r["relative_deviation"]) < 3e-3
    assert r["pair_over_order"] == pytest.approx(32.45 / 74.24, rel=2e-3)
    assert r["uncertainty"] > 0
