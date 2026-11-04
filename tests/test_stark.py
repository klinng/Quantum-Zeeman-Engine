"""Tests for src/stark.py (hydrogen n = 2 in electric and magnetic fields)."""
import numpy as np
import pytest

from src.hydrogen import h_n2, ev_to_mhz
from src.stark import (A_2P_1S, MHZ_PER_A0_VCM, anticrossing_gap_mhz, dipole_operators, h_n2_stark, lifetime_s,
                       perturbative_s_shift_mhz, quenching_rate)


def test_dipole_operators_are_hermitian_with_textbook_size():
    x, z = dipole_operators()
    assert np.allclose(x, x.conj().T) and np.allclose(z, z.conj().T)
    assert z[0, 4] == pytest.approx(-3.0)                  # <2s|z|2p, m=0> = -3 a0
    assert abs(x[0, 2]) == pytest.approx(3 / np.sqrt(2))   # <2s|x|2p, m=+-1> = 3/sqrt(2) a0


def test_zero_electric_field_reproduces_the_magnetic_only_model():
    assert np.allclose(h_n2_stark(0.03, 0.0), h_n2(0.03))


def test_hamiltonian_is_hermitian():
    h = h_n2_stark(0.05, 12.0, 37.0)
    assert np.allclose(h, h.conj().T)


def test_eigenvalues_do_not_depend_on_field_direction_at_zero_b():
    # at B = 0 space is isotropic: the spectrum cannot depend on the angle between E and the quantisation axis
    e0 = np.linalg.eigvalsh(h_n2_stark(1e-9, 15.0, 0.0))
    e1 = np.linalg.eigvalsh(h_n2_stark(1e-9, 15.0, 90.0))
    assert ev_to_mhz(np.abs(e0 - e1)).max() < 1e-3


def test_quadratic_stark_shift_of_2s_matches_perturbation_theory():
    shifts = ev_to_mhz(np.linalg.eigvalsh(h_n2_stark(1e-6, 10.0, 0.0)) - np.linalg.eigvalsh(h_n2_stark(1e-6, 0.0, 0.0)))
    assert shifts.max() > 0
    assert np.isclose(shifts, perturbative_s_shift_mhz(10.0), rtol=0.02).any()


def test_magnetic_field_alone_crosses_but_perpendicular_e_opens_an_anticrossing():
    assert anticrossing_gap_mhz(0.0)[0] < 0.5                     # no electric field: levels cross
    gap, b = anticrossing_gap_mhz(10.0)
    assert gap > 30.0
    assert b * 1e4 == pytest.approx(573.3, abs=3.0)


def test_gap_is_linear_in_e_and_has_the_expected_size():
    g5, g20 = anticrossing_gap_mhz(5.0)[0], anticrossing_gap_mhz(20.0)[0]
    assert g20 / g5 == pytest.approx(4.0, rel=0.02)
    assert g5 / 5.0 == pytest.approx(2 * np.sqrt(3) * MHZ_PER_A0_VCM, rel=0.05)    # 2 |V|, |V| = sqrt(3) a0 e E


def test_parallel_e_does_not_open_the_573_gauss_gap():
    # E parallel to B keeps m_j: 2S(-1/2) and 2P1/2(+1/2) differ in m_j, so they still cross
    bs = np.linspace(0.054, 0.060, 241)
    gaps = [np.diff(ev_to_mhz(np.linalg.eigvalsh(h_n2_stark(b, 10.0, 0.0)))).min() for b in bs]
    assert min(gaps) < 0.5


def test_quenching_rate_is_zero_for_pure_2s_and_grows_with_field():
    # eigenstates are in ascending energy: 0, 1 = 2P1/2, 2, 3 = 2S1/2, 4 to 7 = 2P3/2
    r0 = quenching_rate(1e-6, 0.0, 0.0)
    assert r0[2:4] == pytest.approx([0.0, 0.0], abs=1e-6 * A_2P_1S)      # a pure 2S state has no E1 decay
    assert r0[0] == pytest.approx(A_2P_1S, rel=1e-6)                       # a pure 2P state decays at the 2P rate
    low, high = quenching_rate(1e-6, 10.0, 0.0)[2], quenching_rate(1e-6, 50.0, 0.0)[2]
    assert 0 < low < high


def test_lifetime_limits():
    assert lifetime_s(0.0) == pytest.approx(0.122, rel=0.01)
    assert lifetime_s(A_2P_1S) == pytest.approx(1.6e-9, rel=0.01)
