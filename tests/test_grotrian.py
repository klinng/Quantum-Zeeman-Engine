"""Tests for src/optics/grotrian.py (level-and-arrow diagram of Zeeman transitions)."""
import numpy as np
import pytest

from src.optics.grotrian import fig_grotrian, grotrian_data


@pytest.mark.parametrize("b", [0.05, 1.0, 5.0, 25.0, 60.0])
def test_every_arrow_obeys_the_selection_rule(b):
    d = grotrian_data(b)
    a = d["arrows"]
    dm = d["m_up"][a.upper.values] - d["m_low"][a.lower.values]
    assert np.allclose(dm, a.q.values, atol=1e-6)


def test_arrow_polarisation_matches_q():
    a = grotrian_data(1.0)["arrows"]
    assert set(a[a.q == 0].polarisation) == {"pi"}
    assert set(a[a.q == 1].polarisation) == {"sigma+"}
    assert set(a[a.q == -1].polarisation) == {"sigma-"}


def test_m_values_are_the_expected_half_integers():
    d = grotrian_data(3.0)
    assert np.allclose(np.sort(d["m_up"]), [-1.5, -0.5, -0.5, 0.5, 0.5, 1.5], atol=1e-6)
    assert np.allclose(np.sort(d["m_low"]), [-0.5, 0.5], atol=1e-6)


def test_ten_allowed_arrows_and_strong_ones_fade_in_paschen_back():
    weak, strong = grotrian_data(0.1)["arrows"], grotrian_data(60.0)["arrows"]
    assert len(weak) == 10                                   # 4 (D1) + 6 (D2)
    assert (weak.intensity > 0.2).sum() == 10
    assert (strong.intensity > 0.2).sum() == 6               # four spin-flip lines fade away at 60 T
    assert strong.intensity.min() < 0.05


def test_no_arrow_where_the_matrix_element_vanishes():
    a = grotrian_data(1.0, min_fraction=1e-6)["arrows"]
    assert (a.intensity > 0).all()
    assert len(a) <= 12 * 1                                  # 6 upper x 2 lower states: at most 12 pairs


def test_each_upper_and_lower_state_pair_appears_once_per_polarisation():
    a = grotrian_data(2.0)["arrows"]
    assert not a.duplicated(["upper", "lower"]).any()


def test_figure_builds():
    import matplotlib
    matplotlib.use("Agg")
    assert fig_grotrian((0.5, 10.0)) is not None
