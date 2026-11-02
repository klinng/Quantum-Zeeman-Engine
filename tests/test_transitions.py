"""Level 4: Wigner symbols, selection rules, and the sodium D-line Zeeman patterns."""
import numpy as np
import pytest

from src.constants import MU_B_EV, wavenumber_to_ev
from src.transitions import dipole_matrices, wigner_3j, zeeman_lines
from src.validation import zeta_from_fine_structure

ZETA = zeta_from_fine_structure(17.196, l=1)
D1_CM1 = 16956.17
GAP = wavenumber_to_ev(D1_CM1) + ZETA          # 2P1/2 lies at E0 - zeta
D1_CENTRE, D2_CENTRE = GAP - ZETA, GAP + ZETA / 2
B = 0.05                                       # tesla: weak field, mu_B B << zeta


@pytest.fixture(scope="module")
def lines():
    df = zeeman_lines(1, 0, B, ZETA, 0.0, GAP)
    mid = 0.5 * (D1_CENTRE + D2_CENTRE)
    return df[df.energy_ev < mid], df[df.energy_ev >= mid]


def test_wigner_3j_known_values():
    assert wigner_3j(1, 1, 0, 0, 0, 0) == pytest.approx(-1 / np.sqrt(3))
    assert wigner_3j(1, 1, 2, 0, 0, 0) == pytest.approx(np.sqrt(2 / 15))
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0          # parity: l+l'+1 must be even... here odd sum
    assert wigner_3j(1, 1, 0, 1, 0, 0) == 0.0          # m sum rule


def test_selection_rule_structure():
    """Non-zero only when m_up - m_low = q (so |Delta m| <= 1) AND the spin is unchanged."""
    d = dipole_matrices(1, 0)
    ml_up = [1, 1, 0, 0, -1, -1]
    spin_up = [0, 1, 0, 1, 0, 1]            # index 0 = m_s +1/2, 1 = m_s -1/2
    for q, mat in d.items():
        for i, (mu, su) in enumerate(zip(ml_up, spin_up)):
            for j in range(2):                 # lower index j = spin of the s state
                assert (mat[i, j] != 0) == (mu - 0 == q and su == j)


def test_parity_forbids_s_to_s():
    """l = 0 -> l' = 0 is forbidden: the dipole operator has odd parity."""
    assert all(np.allclose(m, 0) for m in dipole_matrices(0, 0).values())


def test_line_counts(lines):
    d1, d2 = lines
    assert len(d1) == 4 and len(d2) == 6


def test_zeeman_shift_patterns(lines):
    d1, d2 = lines
    s1 = np.sort((d1.energy_ev.to_numpy() - D1_CENTRE) / (MU_B_EV * B))
    s2 = np.sort((d2.energy_ev.to_numpy() - D2_CENTRE) / (MU_B_EV * B))
    assert s1 == pytest.approx([-4 / 3, -2 / 3, 2 / 3, 4 / 3], abs=0.01)
    assert s2 == pytest.approx([-5 / 3, -1, -1 / 3, 1 / 3, 1, 5 / 3], abs=0.01)


def test_polarisations(lines):
    d1, d2 = lines
    assert sorted(d1.polarisation) == ["pi", "pi", "sigma+", "sigma-"]
    assert sorted(d2.polarisation) == ["pi", "pi", "sigma+", "sigma+", "sigma-", "sigma-"]


def test_d2_to_d1_intensity_ratio_is_two(lines):
    d1, d2 = lines
    assert d2.intensity.sum() / d1.intensity.sum() == pytest.approx(2.0, rel=1e-3)


def test_relative_intensities_within_lines(lines):
    d1, d2 = lines
    pi1 = d1[d1.polarisation == "pi"].intensity.to_numpy()
    sig1 = d1[d1.polarisation != "pi"].intensity.to_numpy()
    assert pi1 == pytest.approx([1 / 3] * 2, rel=1e-2)
    assert sig1 == pytest.approx([2 / 3] * 2, rel=1e-2)
    by_shift = d2.assign(shift=(d2.energy_ev - D2_CENTRE) / (MU_B_EV * B)).sort_values("shift")
    assert by_shift.intensity.to_numpy() == pytest.approx([1 / 3, 1, 2 / 3, 2 / 3, 1, 1 / 3], rel=1e-2)


def test_each_upper_state_emits_equally():
    """Wigner-Eckart consequence: all p states have the same total emission rate to the s level."""
    df = zeeman_lines(1, 0, 3.0, ZETA, 0.0, GAP)
    assert df.groupby("upper").intensity.sum().to_numpy() == pytest.approx(np.ones(6), rel=1e-9)


def test_normal_zeeman_triplet():
    """With zeta = 0 and g_s = 0 (orbital only) the pattern is the Lorentz triplet 0, +/- mu_B B."""
    df = zeeman_lines(1, 0, 1.0, 0.0, 0.0, 0.0, g_s=0.0)
    shifts = np.unique(np.round(df.energy_ev / MU_B_EV, 8))
    assert shifts == pytest.approx([-1.0, 0.0, 1.0])
