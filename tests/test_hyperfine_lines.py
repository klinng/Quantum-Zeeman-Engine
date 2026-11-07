import numpy as np
import pytest

from src.hyperfine import NA_A_MHZ
from src.hyperfine_lines import (broadened_spectrum, doppler_fwhm_mhz, group_lines, sodium_d_hyperfine_lines)


@pytest.fixture(scope="module")
def zero_field():
    d = sodium_d_hyperfine_lines(0.0)
    return group_lines(d, "D2"), group_lines(d, "D1"), d


def test_number_of_zero_field_lines(zero_field):
    g2, g1, _ = zero_field
    assert len(g2) == 6 and len(g1) == 4


def test_d2_hyperfine_strengths_match_racah_values(zero_field):
    """F=2 -> F'=1,2,3 share 5 x 2/3 with weights 1/20, 1/4, 7/10; F=1 -> F'=0,1,2 share 3 x 2/3 with 1/6, 5/12, 5/12."""
    g2 = zero_field[0].strength.to_numpy()
    f2 = 5 * 2 / 3 * np.array([1 / 20, 1 / 4, 7 / 10])
    f1 = 3 * 2 / 3 * np.array([1 / 6, 5 / 12, 5 / 12])
    assert np.allclose(g2[:3], f2, atol=1e-6)
    assert np.allclose(sorted(g2[3:]), sorted(f1), atol=1e-6)


def test_d2_to_d1_strength_ratio_is_two(zero_field):
    d = zero_field[2]
    assert d[d.line == "D2"].strength.sum() / d[d.line == "D1"].strength.sum() == pytest.approx(2.0, rel=1e-6)


def test_line_spacings_come_from_the_hyperfine_constants(zero_field):
    g2 = zero_field[0].freq_mhz.to_numpy()
    assert g2[2] - g2[1] == pytest.approx(3 * NA_A_MHZ["3P3/2"], abs=1e-3)       # F'=3 minus F'=2
    assert g2[5] - g2[1] == pytest.approx(1771.6261288, abs=1e-3)                # F=1 -> F'=2 minus F=2 -> F'=2


def test_d2_total_strength_is_field_independent():
    for b in (0.0, 0.01, 0.1, 0.3):
        d = sodium_d_hyperfine_lines(b)
        assert d[d.line == "D2"].strength.sum() == pytest.approx(8 * 2 / 3, rel=1e-6)


def test_doppler_width_scales_as_root_temperature():
    assert doppler_fwhm_mhz(1200.0) == pytest.approx(2 * doppler_fwhm_mhz(300.0), rel=1e-9)


def test_broadened_spectrum_area_equals_total_strength(zero_field):
    d = zero_field[2]
    f = np.linspace(514000 - 5000, 517500 + 5000, 20000)
    area = np.trapezoid(broadened_spectrum(d, "D2", 300.0, f), f)
    assert area == pytest.approx(d[d.line == "D2"].strength.sum(), rel=1e-3)
