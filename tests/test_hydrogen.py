"""Hydrogen n = 2: block structure, Landé slopes, Lamb-shift level scheme, level crossings."""
import numpy as np
import pytest

from src.constants import G_S, MU_B_EV
from src.hamiltonians import landé_g
from src.hydrogen import (FINE_2P_MHZ, LAMB_SHIFT_MHZ, dirac_fine_structure_mhz, ev_to_mhz, h_n2,
                          label_states, level_index, mhz_to_ev, track_n2, _operators)
from src.solvers import find_crossings

B = np.linspace(1e-6, 0.6, 3001)


@pytest.fixture(scope="module")
def sweep():
    return track_n2(B)


def test_zero_field_level_scheme():
    w = np.linalg.eigvalsh(h_n2(1e-8))
    mhz = ev_to_mhz(w)
    assert mhz[:2] == pytest.approx([0, 0], abs=1e-3)                           # 2P1/2
    assert mhz[2:4] == pytest.approx([LAMB_SHIFT_MHZ] * 2, abs=1e-3)            # 2S1/2
    assert mhz[4:] == pytest.approx([FINE_2P_MHZ] * 4, abs=1e-3)                # 2P3/2


def test_dirac_cross_check_of_fine_structure_input():
    """Measured interval vs pure Dirac theory: agree to about 0.2 % (QED corrections)."""
    assert dirac_fine_structure_mhz() == pytest.approx(FINE_2P_MHZ, rel=3e-3)


def test_labels_have_right_multiplicities(sweep):
    _, _, labels, _ = sweep
    assert labels.term.value_counts().to_dict() == {"2P3/2": 4, "2S1/2": 2, "2P1/2": 2}
    assert sorted(labels[labels.term == "2P3/2"].mj) == [-1.5, -0.5, 0.5, 1.5]
    assert sorted(labels[labels.term == "2S1/2"].mj) == [-0.5, 0.5]


def test_weak_field_slopes_equal_landé(sweep):
    """dE/dB = g_J m_j mu_B for every level (2s and 2p separately, they do not mix)."""
    b, energies, lab, _ = track_n2([1e-6, 1e-4, 2e-4])      # tracked: same column = same level
    g = {"2S1/2": landé_g(0, 0.5), "2P1/2": landé_g(1, 0.5), "2P3/2": landé_g(1, 1.5)}
    for _, row in lab.iterrows():
        slope = (energies[2, row.level] - energies[1, row.level]) / (MU_B_EV * 1e-4)
        assert slope == pytest.approx(g[row.term] * row.mj, abs=2e-3)


def test_magnetic_field_never_mixes_2s_with_2p(sweep):
    """l is conserved, so every eigenvector stays purely 2s or purely 2p at every field."""
    _, _, _, vectors = sweep
    p_s = _operators()["P_s"]
    w = np.einsum("bik,ij,bjk->bk", vectors.conj(), p_s, vectors).real
    assert np.all((w < 1e-9) | (w > 1 - 1e-9))


def test_lamb_retherford_style_crossing_near_570_gauss(sweep):
    """2S1/2(mj=-1/2) crosses 2P1/2(mj=+1/2). Literature quotes 'about 570 G' (hyperfine included there)."""
    b, energies, labels, _ = sweep
    i = level_index(labels, "2S1/2", -0.5)
    j = level_index(labels, "2P1/2", 0.5)
    crossings = find_crossings(b, energies[:, i], energies[:, j])
    assert len(crossings) == 1
    b_cross = crossings[0]
    # First-order (straight-line) estimate. The exact crossing is ~1 % higher because the field
    # mixes 2P1/2 with 2P3/2, which bends the 2P1/2 level down by about 13 MHz there.
    linear = mhz_to_ev(LAMB_SHIFT_MHZ) / (MU_B_EV * (G_S * 0.5 + landé_g(1, 0.5) * 0.5))
    assert b_cross == pytest.approx(linear, rel=0.02)
    assert b_cross > linear                                              # curvature pushes it up
    assert b_cross * 1e4 == pytest.approx(570, rel=0.02)                 # literature "about 570 G"
    # independent check: at that field two exact eigenvalues really coincide
    mhz = np.sort(ev_to_mhz(np.linalg.eigvalsh(h_n2(b_cross))))
    assert np.min(np.diff(mhz)) < 0.3


def test_strong_field_slopes_are_paschen_back(sweep):
    """At very large B the 8 slopes become m_l + g_s m_s (2s: +/- g_s/2; 2p: six values)."""
    hi = np.array([50.0, 51.0])
    e = np.array([np.linalg.eigvalsh(h_n2(x)) for x in hi])
    slopes = np.sort((e[1] - e[0]) / (MU_B_EV * (hi[1] - hi[0])))
    expected = np.sort([ml + G_S * ms for ml in (1, 0, -1) for ms in (0.5, -0.5)] + [G_S / 2, -G_S / 2])
    assert slopes == pytest.approx(expected, abs=5e-3)


def test_label_states_on_known_vector():
    """A pure |2s, ms=+1/2> vector is labelled 2S1/2 with m_j = +1/2."""
    v = np.zeros((8, 1), dtype=complex)
    v[0, 0] = 1
    row = label_states(v).iloc[0]
    assert row.term == "2S1/2" and row.mj == 0.5
