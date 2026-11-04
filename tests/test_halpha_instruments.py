"""Tests for src/optics/halpha_instruments.py."""
import pytest

from src.optics.halpha_instruments import (doppler_limited_power, fig_halpha_instrument_ladder, halpha_instrument_ladder,
                                           prism_zeeman_deflection_urad, required_power_halpha, weakest_field_halpha)


def test_required_power_scales_as_inverse_field():
    assert required_power_halpha(1.0) == pytest.approx(32635, rel=1e-3)
    assert required_power_halpha(2.0) == pytest.approx(required_power_halpha(1.0) / 2, rel=1e-12)


def test_weakest_field_is_the_inverse_function():
    for r in (1e3, 3e4, 1e6):
        assert required_power_halpha(weakest_field_halpha(r)) == pytest.approx(r, rel=1e-9)


def test_ladder_is_increasing():
    v = list(halpha_instrument_ladder().values())
    assert v == sorted(v)


def test_doppler_floor_is_about_81000_and_cooler_gas_raises_it():
    assert doppler_limited_power(300.0) == pytest.approx(81000, rel=0.01)
    assert doppler_limited_power(30.0) > doppler_limited_power(300.0)


def test_prism_cannot_separate_groups_at_one_tesla_but_a_large_grating_can():
    ladder = halpha_instrument_ladder()
    need = required_power_halpha(1.0)
    assert ladder["Prism (SF11, 5 cm base)"] < need < ladder["Large grating (1200 l/mm, 10 cm)"]


def test_prism_deflection_grows_with_field_and_sf11_beats_bk7():
    assert prism_zeeman_deflection_urad(6.0) == pytest.approx(6 * prism_zeeman_deflection_urad(1.0), rel=0.01)
    assert prism_zeeman_deflection_urad(1.0, "SF11") > prism_zeeman_deflection_urad(1.0, "BK7")


def test_figure_builds():
    import matplotlib
    matplotlib.use("Agg")
    assert fig_halpha_instrument_ladder() is not None
