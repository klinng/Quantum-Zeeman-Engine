"""The sodium D lines as seen through the virtual spectrometer, against textbook patterns."""
import numpy as np
import pytest

from src.optics.polarization import circular_polariser, linear_polariser
from src.optics.prism import prism_resolving_power
from src.optics.spectrometer import (LAMP_LINES, bandpass_transmission, field_for_resolving_power,
                                     gaussian_spectrum, observe, required_resolving_power,
                                     sodium_d_components, vacuum_cm1_to_air_nm)

B = 0.05    # weak field


@pytest.fixture(scope="module")
def comps():
    return sodium_d_components(B)


def test_zero_field_wavelengths_match_known_air_values():
    assert float(vacuum_cm1_to_air_nm(16973.37)) == pytest.approx(588.995, abs=0.003)   # D2
    assert float(vacuum_cm1_to_air_nm(16956.17)) == pytest.approx(589.592, abs=0.003)   # D1


def test_component_counts(comps):
    assert (comps.line == "D1").sum() == 4 and (comps.line == "D2").sum() == 6


def test_zeeman_shift_scale_about_27_pm_per_tesla_for_outer_d2_lines():
    c = sodium_d_components(1.0)
    outer = c[c.line == "D2"].shift_pm.abs().max()
    assert outer == pytest.approx(27.0, rel=0.04)            # (5/3) mu_B B / hc at 589 nm


def test_transverse_view_d1_has_four_equally_bright_lines(comps):
    """Classic result: viewed across B, the four D1 components have equal intensity."""
    d1 = observe(comps, 90.0)
    d1 = d1[d1.line == "D1"].observed.to_numpy()
    assert d1 == pytest.approx(np.full(4, d1.mean()), rel=1e-2)


def test_transverse_view_d2_intensities_are_4_3_1(comps):
    """Classic result: viewed across B, the D2 components have relative intensities 4 : 3 : 1."""
    d2 = observe(comps, 90.0)
    d2 = d2[d2.line == "D2"].assign(a=lambda d: d.shift_pm.abs().round(-1))
    # group by |shift| ordering: pi (smallest), sigma(+-1), sigma(+-5/3)
    d2 = d2.assign(absshift=d2.shift_pm.abs()).sort_values("absshift")
    pi, s1, s2 = d2.observed.to_numpy()[[0, 2, 4]]
    assert [pi / s2, s1 / s2] == pytest.approx([4.0, 3.0], rel=2e-2)


def test_longitudinal_view_has_no_pi_lines(comps):
    seen = observe(comps, 0.0)
    pi = seen[seen.polarisation == "pi"].observed
    sig = seen[seen.polarisation != "pi"].observed
    assert pi.max() < 1e-12
    assert (sig > 0.1).all()
    assert (seen.observed > 1e-9).sum() == 6            # 2 (D1) + 4 (D2) sigma lines


def test_polariser_parallel_to_b_keeps_only_pi_in_transverse_view(comps):
    seen = observe(comps, 90.0, linear_polariser(0))
    assert seen[seen.polarisation != "pi"].observed.max() < 1e-12
    assert (seen[seen.polarisation == "pi"].observed > 0.1).all()


def test_polariser_perpendicular_to_b_keeps_only_sigma_in_transverse_view(comps):
    seen = observe(comps, 90.0, linear_polariser(90))
    assert seen[seen.polarisation == "pi"].observed.max() < 1e-12
    assert (seen[seen.polarisation != "pi"].observed > 0.05).all()


def test_circular_analyser_separates_sigma_plus_and_minus(comps):
    plus = observe(comps, 0.0, circular_polariser(+1))
    minus = observe(comps, 0.0, circular_polariser(-1))
    assert plus[plus.polarisation == "sigma-"].observed.max() < 1e-12
    assert minus[minus.polarisation == "sigma+"].observed.max() < 1e-12
    assert plus[plus.polarisation == "sigma+"].observed.min() > 0.1
    assert minus[minus.polarisation == "sigma-"].observed.min() > 0.1


def test_total_observed_intensity_conserved_for_unpolarised_detection(comps):
    """Without an element the observed total at 90 deg is sum(pi) + sum(sigma)/2."""
    seen = observe(comps, 90.0)
    expected = comps[comps.polarisation == "pi"].rate.sum() + 0.5 * comps[comps.polarisation != "pi"].rate.sum()
    assert seen.observed.sum() == pytest.approx(expected, rel=1e-9)


def test_prism_cannot_resolve_zeeman_splitting_at_one_tesla():
    c = sodium_d_components(1.0)
    needed = required_resolving_power(c)
    assert needed > 30000
    assert prism_resolving_power(589.3, 50, "BK7") < needed / 10
    # the field that a 5 cm BK7 prism could resolve is far beyond the weak-field regime (B* = 24.6 T)
    assert field_for_resolving_power(prism_resolving_power(589.3, 50, "BK7")) > 15


def test_lamp_table_and_filters():
    assert len(LAMP_LINES) == 11 and LAMP_LINES.wavelength_nm.between(380, 750).all()
    assert float(bandpass_transmission(550, 550, 10)) == pytest.approx(1.0)
    assert float(bandpass_transmission(555, 550, 10)) == pytest.approx(0.5)      # half max at +FWHM/2
    wl = np.linspace(540, 560, 4001)
    y = gaussian_spectrum(wl, [550.0], [1.0], 4.0)
    above = wl[y >= 0.5]
    assert above.max() - above.min() == pytest.approx(4.0, abs=0.01)


def test_requires_positive_field():
    with pytest.raises(ValueError):
        sodium_d_components(0.0)
