"""Instrument ladder: resolving-power formulas and the field each instrument needs."""
import numpy as np
import pytest

from src.optics.instruments import (B_STAR_TESLA, fabry_perot_resolving_power, fig_instrument_ladder,
                                    grating_resolving_power, instrument_ladder, required_power_vs_field,
                                    zeeman_shift_nm)
from src.optics.spectrometer import field_for_resolving_power, required_resolving_power, sodium_d_components


def test_grating_formula():
    assert grating_resolving_power(1200, 100) == 120000
    assert grating_resolving_power(600, 50, order=2) == 60000


def test_fabry_perot_formula():
    m = 2 * 10e6 / 589.3                                           # interference order for a 10 mm gap
    assert fabry_perot_resolving_power(10, 30, 589.3) == pytest.approx(m * 30)
    assert fabry_perot_resolving_power(20, 30) == pytest.approx(2 * fabry_perot_resolving_power(10, 30))


def test_ladder_is_ordered_and_spans_orders_of_magnitude():
    r = list(instrument_ladder().values())
    assert r == sorted(r)
    assert r[-1] / r[0] > 100


def test_required_power_falls_as_one_over_field():
    r = required_power_vs_field([0.25, 0.5, 1.0, 2.0])
    assert r[0] / r[1] == pytest.approx(2.0, rel=0.03)
    assert r[1] / r[2] == pytest.approx(2.0, rel=0.03)
    assert r[2] / r[3] == pytest.approx(2.0, rel=0.03)


def test_field_for_resolving_power_is_consistent_with_required_power():
    for b in (0.1, 0.5, 1.0):
        r = required_resolving_power(sodium_d_components(b))
        assert field_for_resolving_power(r) == pytest.approx(b, rel=0.05)


def test_a_prism_needs_a_field_near_the_end_of_the_weak_field_regime():
    prism = instrument_ladder()["Prism (BK7, 5 cm base)"]
    assert field_for_resolving_power(prism) > 0.8 * B_STAR_TESLA          # cannot show Zeeman components
    assert field_for_resolving_power(instrument_ladder()["Large grating (1200 l/mm, 10 cm)"]) < 1.0


def test_outermost_d2_component_shift_is_27_pm_per_tesla():
    assert zeeman_shift_nm(1.0) * 1000 == pytest.approx(27.0, rel=0.03)


def test_ladder_figure_builds():
    import matplotlib
    matplotlib.use("Agg")
    fig = fig_instrument_ladder()
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)
