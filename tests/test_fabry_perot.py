"""Tests for src/optics/fabry_perot.py."""
import numpy as np
import pytest
from scipy.signal import find_peaks

from src.optics.fabry_perot import _d2, airy, radial_profile, reflectivity_for_finesse, ring_angle
from src.optics.polarization import linear_polariser
from src.optics.spectrometer import observe

GAP, FIN = 5.0, 30.0


def _peaks(b, element):
    refl = reflectivity_for_finesse(FIN)
    d = observe(_d2(b), 90.0, element)
    d = d[d.observed > 1e-3 * d.observed.max()]
    th0 = ring_angle(2, GAP)
    th = np.linspace(th0 - 1.5e-3, th0 + 1.5e-3, 6000)
    prof = radial_profile(th, d, GAP, refl)
    idx, _ = find_peaks(prof, height=0.05 * prof.max(), prominence=0.02 * prof.max())
    return prof[idx]


def test_finesse_formula_round_trip():
    r = reflectivity_for_finesse(FIN)
    assert np.pi * np.sqrt(r) / (1 - r) == pytest.approx(FIN, rel=1e-9)


def test_airy_peaks_at_resonance_and_is_small_between():
    lam = 588.995
    m = 2 * GAP * 1e6 / lam
    th = ring_angle(1, GAP, lam)
    assert airy(th, lam, GAP, reflectivity_for_finesse(FIN)) == pytest.approx(1.0, abs=1e-3)
    th_mid = np.arccos((np.floor(m) - 1.5) / m)
    assert airy(th_mid, lam, GAP, reflectivity_for_finesse(FIN)) < 0.01


def test_zero_field_ring_is_single():
    assert len(_peaks(1e-3, None)) == 1                                   # six components within 0.03 pm: one ring


def test_fabry_perot_resolves_six_components_at_0p3_tesla():
    assert len(_peaks(0.3, None)) == 6


def test_perpendicular_polariser_leaves_four_sigma_peaks():
    assert len(_peaks(0.3, linear_polariser(90.0))) == 4


def test_parallel_polariser_leaves_two_pi_peaks():
    assert len(_peaks(0.3, linear_polariser(0.0))) == 2


def test_peak_heights_follow_the_4_3_1_pattern_roughly():
    h = np.sort(_peaks(0.3, None))
    # outer sigma (2 weakest), inner sigma (2 middle), pi (2 strongest): ratios near 1 : 3 : 4
    assert h[2:4].mean() / h[:2].mean() == pytest.approx(3.0, rel=0.25)
    assert h[4:].mean() / h[:2].mean() == pytest.approx(4.0, rel=0.25)
