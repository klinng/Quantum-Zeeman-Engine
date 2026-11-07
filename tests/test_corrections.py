import numpy as np
import pytest
from scipy import constants as sc
from scipy.integrate import quad

from src.diamagnetic import (balmer_alpha_diamagnetic_shift_cm1, diamagnetic_shift_ev, r2_hydrogenic, sin2_theta_mean)
from src.hydrogen_spectrum import radial_function
from src.isotope import isotope_shift, line_wavenumber_cm1, rydberg_cm1, zeeman_shift_pm_per_t


def test_diamagnetic_ground_state_matches_atomic_unit_result():
    """For 1s the first-order shift is B^2/4 in atomic units (B in units of 2.35e5 T, energy in hartree)."""
    b_au = 1.0
    b_t = b_au * sc.hbar / (sc.e * sc.physical_constants["Bohr radius"][0] ** 2)
    hartree_ev = sc.physical_constants["Hartree energy in eV"][0]
    assert diamagnetic_shift_ev(1, 0, 0, b_t) == pytest.approx(0.25 * hartree_ev, rel=1e-9)


@pytest.mark.parametrize("n,l", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_r2_formula_matches_numerical_integral(n, l):
    num = quad(lambda r: radial_function(n, l, r) ** 2 * r ** 4, 0, 80 * n * n)[0]
    assert r2_hydrogenic(n, l) == pytest.approx(num, rel=1e-6)


def test_angular_average_sums_over_m():
    for l in (1, 2):
        assert np.mean([sin2_theta_mean(l, m) for m in range(-l, l + 1)]) == pytest.approx(2 / 3, abs=1e-12)


def test_quadratic_in_field_and_small_against_zeeman_at_60t():
    a, b = balmer_alpha_diamagnetic_shift_cm1(20.0), balmer_alpha_diamagnetic_shift_cm1(60.0)
    assert b["max"] == pytest.approx(9 * a["max"], rel=1e-9)
    zeeman_cm1 = sc.physical_constants["Bohr magneton in inverse meters per tesla"][0] / 100 * 60
    assert b["max"] / zeeman_cm1 < 0.01


def test_isotope_shift_sign_and_size():
    s = isotope_shift()
    assert s["wavenumber_cm1"] > 0 and s["wavelength_pm"] < 0                 # D is bluer
    assert 170 < -s["wavelength_pm"] < 190                                     # about 0.18 nm
    assert rydberg_cm1("D") > rydberg_cm1("H")


def test_zeeman_scale_nearly_isotope_independent():
    assert zeeman_shift_pm_per_t("D") == pytest.approx(zeeman_shift_pm_per_t("H"), rel=1e-3)
    assert zeeman_shift_pm_per_t("H") == pytest.approx(20.1, abs=0.1)


def test_corrections_figure_uses_the_8pm_doppler_width():
    """Regression: an earlier call passed the temperature as the wavelength and plotted 3.7 pm."""
    from src.hydrogen_spectrum import doppler_fwhm_pm
    assert doppler_fwhm_pm(temperature_k=300.0) == pytest.approx(8.1, abs=0.05)
    import inspect, src.corrections_figures as m
    assert "doppler_fwhm_pm(300.0)" not in inspect.getsource(m)
