import numpy as np
import pytest

from src.line_profiles import (NA_A_D2, natural_fwhm_mhz, voigt_fwhm_approx, voigt_fwhm_exact, voigt_mhz,
                               voigt_spectrum)
from src.hyperfine_lines import broadened_spectrum, sodium_d_hyperfine_lines


def test_na_d2_natural_width():
    assert natural_fwhm_mhz(NA_A_D2) == pytest.approx(9.80, abs=0.01)          # 0.616e8 / (2 pi)


def test_unit_area():
    f = np.linspace(-40000, 40000, 1600001)          # Lorentzian tails are long: 2*gamma/(pi*L) is lost beyond +-L
    assert np.trapezoid(voigt_mhz(f, 0.0, 100.0, 30.0), f) == pytest.approx(1.0, abs=1e-3)


def test_limits_gaussian_and_lorentzian():
    f = np.linspace(-500, 500, 11)
    gauss = np.exp(-0.5 * (f / (100 / (2 * np.sqrt(2 * np.log(2))))) ** 2) / ((100 / (2 * np.sqrt(2 * np.log(2)))) * np.sqrt(2 * np.pi))
    assert np.allclose(voigt_mhz(f, 0.0, 100.0, 1e-9), gauss, rtol=1e-4)
    lor = (15.0 / np.pi) / (f ** 2 + 15.0 ** 2)
    assert np.allclose(voigt_mhz(f, 0.0, 0.0, 30.0), lor, rtol=1e-12)


def test_fwhm_pure_cases_and_approximation():
    assert voigt_fwhm_exact(100.0, 1e-6) == pytest.approx(100.0, rel=1e-4)
    assert voigt_fwhm_exact(1e-6, 40.0) == pytest.approx(40.0, rel=1e-4)
    for fg, fl in ((100, 10), (10, 100), (50, 50)):
        assert voigt_fwhm_approx(fg, fl) == pytest.approx(voigt_fwhm_exact(fg, fl), rel=5e-3)


def test_voigt_spectrum_area_equals_strength():
    d = sodium_d_hyperfine_lines(0.0)
    f = np.linspace(513000, 519000, 120001)
    area = np.trapezoid(voigt_spectrum(d, "D2", 25.0, 9.8, f), f)
    assert area == pytest.approx(d[d.line == "D2"].strength.sum(), rel=2e-3)


def test_voigt_reduces_to_gaussian_spectrum_when_lorentz_small():
    d = sodium_d_hyperfine_lines(0.0)
    f = np.linspace(514000, 517500, 50)
    assert np.allclose(voigt_spectrum(d, "D2", 1300.0, 1e-9, f), broadened_spectrum(d, "D2", 1300.0, f), rtol=1e-3)
