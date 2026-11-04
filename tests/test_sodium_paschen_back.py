"""Tests for src/optics/sodium_paschen_back.py (sodium D lines from weak field to Paschen-Back)."""
import numpy as np
import pytest

from src.optics.sodium_paschen_back import COG_NM, components_vs_cog

NORMAL_PM_PER_T = 16.2     # lambda^2 * mu_B / (h c) at 589.2 nm


@pytest.mark.parametrize("b", [1.0, 10.0, 60.0, 120.0])
def test_mean_sigma_shift_is_the_normal_zeeman_shift_at_every_field(b):
    d = components_vs_cog(b)
    mean = lambda q: np.average(d[d.q == q].shift_pm, weights=d[d.q == q].rate)
    assert mean(-1) / b == pytest.approx(NORMAL_PM_PER_T, rel=0.02)
    assert mean(1) / b == pytest.approx(-NORMAL_PM_PER_T, rel=0.02)
    assert abs(mean(0)) / b < 0.2


def test_ten_components_at_every_field():
    for b in (1.0, 25.0, 60.0):
        assert len(components_vs_cog(b)) == 10


def test_strong_lines_dominate_at_60_tesla():
    d = components_vs_cog(60.0)
    assert d[d.rate > 0.3].rate.sum() / d.rate.sum() > 0.85        # the doublet has collapsed into a few strong lines
    assert d[d.rate < 0.15].rate.sum() / d.rate.sum() < 0.1        # the leftover weak lines carry little


def test_weak_field_has_no_collapse():
    d = components_vs_cog(1.0)
    assert d[d.rate < 0.15].empty or d[d.rate < 0.15].rate.sum() / d.rate.sum() < 0.1


def test_centre_of_gravity_wavelength():
    assert COG_NM == pytest.approx(589.19, abs=0.02)
