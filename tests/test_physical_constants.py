"""Numerical outputs vs known physics limits."""
import numpy as np
import pytest

from src.constants import G_S, MU_B_EV, ev_to_wavenumber, wavenumber_to_ev
from src.hamiltonians import landé_g
from src.solvers import energy_vs_field
from src.validation import fine_structure_splitting_cm1, load_reference, zeta_from_fine_structure

ZETA_NA = zeta_from_fine_structure(17.196, l=1)   # eV


def test_bohr_magneton():
    assert MU_B_EV == pytest.approx(5.7883818060e-5, rel=1e-9)


def test_unit_roundtrip():
    assert ev_to_wavenumber(wavenumber_to_ev(17.196)) == pytest.approx(17.196)


def test_landé_values():
    assert landé_g(1, 0.5, g_s=2.0) == pytest.approx(2 / 3)
    assert landé_g(1, 1.5, g_s=2.0) == pytest.approx(4 / 3)
    assert landé_g(0, 0.5, g_s=2.0) == pytest.approx(2.0)


def test_fine_structure_reproduces_experiment():
    """If zeta is set from the measured splitting, the solver must give the same splitting back."""
    assert fine_structure_splitting_cm1(1, ZETA_NA) == pytest.approx(17.196, rel=1e-9)


def test_weak_field_slopes_match_landé():
    """At small B the levels split linearly with slope g_J * m_J * mu_B."""
    b = 1e-3 * ZETA_NA / MU_B_EV                      # mu_B B << zeta
    e0 = energy_vs_field(1, [0.0], ZETA_NA)[0]
    e1 = energy_vs_field(1, [b], ZETA_NA)[0]
    shifts = np.sort((e1 - e0) / (MU_B_EV * b))
    g12, g32 = landé_g(1, 0.5), landé_g(1, 1.5)
    expected = np.sort([g12 * m for m in (-0.5, 0.5)] + [g32 * m for m in (-1.5, -0.5, 0.5, 1.5)])
    assert shifts == pytest.approx(expected, rel=2e-2, abs=1e-3)


def test_stretched_states_exact():
    """|m_l=+1, m_s=+1/2> and |m_l=-1, m_s=-1/2> are exact eigenstates for every B."""
    for b_over_zeta in (0.5, 5, 200):
        b = b_over_zeta * ZETA_NA / MU_B_EV
        e = energy_vs_field(1, [b], ZETA_NA)[0]
        top = ZETA_NA / 2 + MU_B_EV * b * (1 + G_S / 2)
        bottom = ZETA_NA / 2 - MU_B_EV * b * (1 + G_S / 2)
        assert e[-1] == pytest.approx(top, rel=1e-10)
        assert np.min(np.abs(e - bottom)) < 1e-10 * abs(bottom)   # not always the lowest at small B


def test_paschen_back_limit():
    """At very large B the energies approach mu_B B (m_l + g_s m_s) + zeta m_l m_s.

    Tolerance is a few percent of zeta: levels with equal m_l + m_s stay nearly degenerate
    (g_s is only 0.0023 above 2), so second-order mixing ~ zeta^2 / (offset) remains.
    """
    b = 1e4 * ZETA_NA / MU_B_EV
    e = energy_vs_field(1, [b], ZETA_NA)[0]
    expected = []
    for ml in (1, 0, -1):
        for ms in (0.5, -0.5):
            expected.append(MU_B_EV * b * (ml + G_S * ms) + ZETA_NA * ml * ms)
    assert e == pytest.approx(np.sort(expected), abs=0.03 * ZETA_NA)


def test_reference_file_loads():
    df = load_reference()
    assert {"quantity", "value", "unit"} <= set(df.columns)
    assert len(df) >= 4
