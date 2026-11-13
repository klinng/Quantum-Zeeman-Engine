import numpy as np
import pytest

from src.constants import G_S, MU_B_EV
from src.hydrogen import ev_to_mhz, h_n2
from src.hyperfine import (H_G_I, H_HFS_2P12_MHZ, H_HFS_2P32_MHZ, H_HFS_2S_MHZ, NA_A_MHZ, NA_G_I, NA_I,
                           h_n2_hyperfine, h_na_3p, h_na_3s, lamb_crossings_gauss)


def test_hermitian():
    for h in (h_na_3s(0.1), h_n2_hyperfine(0.05), h_na_3p(0.2, 0.002)):
        assert np.allclose(h, h.conj().T)


def test_sodium_ground_splitting_matches_published():
    w = ev_to_mhz(np.linalg.eigvalsh(h_na_3s(0.0)))
    assert w.max() - w.min() == pytest.approx(1771.6261288, abs=1e-3)
    assert np.sum(np.isclose(w, w.min(), atol=1e-6)) == 3     # F = 1
    assert np.sum(np.isclose(w, w.max(), atol=1e-6)) == 5     # F = 2


def test_sodium_stretched_state_is_exact():
    """The state m_F = I + 1/2 has E = A I/2 + mu_B B (g_J/2 + g_I I) exactly."""
    b = 0.37
    a = NA_A_MHZ["3S1/2"]
    expected = a * NA_I / 2 * 1.0 + ev_to_mhz(MU_B_EV * b * (G_S / 2 + NA_G_I * NA_I))
    e_max = ev_to_mhz(np.linalg.eigvalsh(h_na_3s(b)).max())
    assert e_max == pytest.approx(expected, abs=1e-3)


def test_sodium_3p_splittings():
    w = np.sort(ev_to_mhz(np.linalg.eigvalsh(h_na_3p(0.0, 0.002))))
    # 3P1/2: F = 1, 2 split by 2A = 188.88 MHz
    p12 = w[:8]
    assert p12.max() - p12.min() == pytest.approx(2 * NA_A_MHZ["3P1/2"], abs=1e-3)


def test_hydrogen_zero_field_hyperfine():
    w = np.sort(ev_to_mhz(np.linalg.eigvalsh(h_n2_hyperfine(0.0))))
    assert w[4:8].max() - w[4:8].min() == pytest.approx(H_HFS_2S_MHZ, abs=1e-4)
    assert (w[0:4].max() - w[0:4].min()) == pytest.approx(H_HFS_2P12_MHZ, abs=1e-4)
    assert (w[8:16].max() - w[8:16].min()) == pytest.approx(H_HFS_2P32_MHZ, abs=1e-4)


def test_proton_g_i_sign_and_size():
    assert H_G_I == pytest.approx(-0.003042, abs=1e-5)


def test_conserved_fz():
    """H commutes with F_z = J_z + I_z in every block (check on sodium 3S)."""
    from src.hamiltonians import angular_momentum_ops
    _, _, jz = angular_momentum_ops(0.5)
    _, _, iz = angular_momentum_ops(1.5)
    fz = np.kron(jz, np.eye(4)) + np.kron(np.eye(2), iz)
    h = h_na_3s(0.4)
    assert np.allclose(h @ fz, fz @ h, atol=1e-12)


def test_lamb_crossing_splits_into_four():
    plain = lamb_crossings_gauss(hyperfine=False)
    hfs = lamb_crossings_gauss(hyperfine=True)
    assert len(plain) == 1 and plain[0] == pytest.approx(573.4, abs=0.2)
    assert len(hfs) == 4
    assert 570.0 < np.mean(hfs) < 573.0
    assert max(hfs) - min(hfs) > 50.0


def test_leading_order_hydrogen_formula_against_known_levels():
    from src.hyperfine import (H_FERMI_1S_MHZ, H_HFS_2P12_MHZ, H_HFS_2P32_MHZ, H_HFS_2S_MHZ, hydrogen_hfs_leading_mhz as f)
    assert f(1, 0, 0.5) == pytest.approx(H_FERMI_1S_MHZ)
    assert f(2, 0, 0.5) == pytest.approx(H_HFS_2S_MHZ, rel=1e-4)          # measured; relativistic correction about 4e-5
    assert f(2, 1, 0.5) == pytest.approx(H_HFS_2P12_MHZ, rel=2e-3)        # theory value; correction about 9e-4
    assert f(2, 1, 1.5) == pytest.approx(H_HFS_2P32_MHZ)                  # the constant used in h_n2_hyperfine
    assert f(3, 1, 0.5) == pytest.approx(f(3, 0, 0.5) / 3)
    assert f(3, 0, 0.5) == pytest.approx(H_FERMI_1S_MHZ / 27)


def test_d21_combination_agrees_with_bound_state_qed():
    """D21 = 8 f(2S) - f(1S) removes the nuclear-structure part; measured 48 959.2(6.8) Hz vs theory 48 954.1(2.3) Hz."""
    from src.hyperfine import D21_THEORY_HZ, D21_THEORY_SIGMA_HZ, H_FERMI_1S_MHZ, H_HFS_2S_MHZ, H_HFS_2S_SIGMA_MHZ
    d21 = (8 * H_HFS_2S_MHZ - H_FERMI_1S_MHZ) * 1e6
    assert d21 == pytest.approx(48959.2, abs=0.2)
    sigma = np.hypot(6.8, D21_THEORY_SIGMA_HZ)
    assert abs(d21 - D21_THEORY_HZ) < sigma                       # about 0.7 sigma


# --- sodium 3P3/2 electric quadrupole (hyperfine audit) -----------------------------------------------------------
from src.hamiltonians import angular_momentum_ops
from src.hydrogen import mhz_to_ev
from src.hyperfine import NA_B_MHZ, f_weights, h_with_hyperfine, quadrupole_operator

_A32, _B32 = NA_A_MHZ["3P3/2"], NA_B_MHZ["3P3/2"]
_ZETA = 0.002                                   # eV, any fine-structure value: only the 3P3/2 block is examined


def _p32_levels(quadrupole: bool) -> np.ndarray:
    """The 16 levels of 3P3/2 (F = 0..3) at zero field, MHz, ascending."""
    return np.sort(ev_to_mhz(np.linalg.eigvalsh(h_na_3p(0.0, _ZETA, quadrupole=quadrupole))))[8:]


def _e_f(f: int, a: float = _A32, b: float = _B32, i: float = 1.5, j: float = 1.5) -> float:
    """Casimir formula: E(F) = A K/2 + B [3/4 K(K+1) - I(I+1)J(J+1)] / [2I(2I-1)J(2J-1)], K = F(F+1) - I(I+1) - J(J+1)."""
    k = f * (f + 1) - i * (i + 1) - j * (j + 1)
    return a * k / 2 + b * (0.75 * k * (k + 1) - i * (i + 1) * j * (j + 1)) / (2 * i * (2 * i - 1) * j * (2 * j - 1))


@pytest.mark.implementation
def test_every_f_level_matches_the_casimir_formula_with_its_degeneracy():
    w = _p32_levels(True)
    expected = np.sort(np.concatenate([[_e_f(f)] * (2 * f + 1) for f in (0, 1, 2, 3)]))
    # the model's zero of energy is the 3P1/2 centre; the pattern must agree up to one common constant
    assert np.allclose(w - expected, (w - expected)[0], atol=1e-5)
    shift = {f: _e_f(f) - _e_f(f, b=0.0) for f in (0, 1, 2, 3)}                      # quadrupole-only shift of each F level
    assert shift[0] == pytest.approx(1.25 * _B32) and shift[1] == pytest.approx(0.25 * _B32)
    assert shift[2] == pytest.approx(-0.75 * _B32) and shift[3] == pytest.approx(0.25 * _B32)
    assert sum((2 * f + 1) * shift[f] for f in shift) == pytest.approx(0.0, abs=1e-12)    # traceless: no net shift
    # the same shifts from the code: with-B minus without-B levels, per F level
    d = _p32_levels(True) - _p32_levels(False)
    assert np.isclose(d, shift[0], atol=1e-5).sum() == 1 and np.isclose(d, shift[3], atol=1e-5).sum() >= 7


@pytest.mark.implementation
def test_3p32_intervals_with_and_without_quadrupole():
    for quad, expected in ((False, [_A32, 2 * _A32, 3 * _A32]), (True, [_A32 - _B32, 2 * _A32 - _B32, 3 * _A32 + _B32])):
        w = np.unique(np.round(_p32_levels(quad), 5))
        assert np.allclose(np.diff(w), expected, atol=1e-4)                           # F' = 1-0, 2-1, 3-2
    w = np.unique(np.round(_p32_levels(True), 5))
    assert np.allclose(np.diff(w), [15.810, 34.344, 58.326], atol=2e-3)               # evaluated from A = 18.534, B = 2.724


@pytest.mark.implementation
def test_b_zero_recovers_the_dipole_only_model():
    a = {0.5: mhz_to_ev(NA_A_MHZ["3P1/2"]), 1.5: mhz_to_ev(NA_A_MHZ["3P3/2"])}
    h0 = h_with_hyperfine(1, NA_I, a, 0.2, NA_G_I, _ZETA, e0_ev=_ZETA, b_quad_by_j_ev={1.5: 0.0})
    assert np.allclose(h0, h_na_3p(0.2, _ZETA, quadrupole=False), atol=1e-14)


@pytest.mark.implementation
def test_quadrupole_vanishes_for_j_half_and_for_i_half():
    big = mhz_to_ev(50.0)
    # I = 1/2 (hydrogen): the operator is identically zero, whatever B_J is
    a = {0.5: mhz_to_ev(59.0), 1.5: mhz_to_ev(11.8)}
    base = h_with_hyperfine(1, 0.5, a, 0.1, -0.003, 0.001)
    assert np.allclose(base, h_with_hyperfine(1, 0.5, a, 0.1, -0.003, 0.001, b_quad_by_j_ev={0.5: big, 1.5: big}), atol=1e-14)
    # J = 1/2 with I = 3/2 (sodium 3S1/2 and 3P1/2): zero as well
    a = {0.5: mhz_to_ev(94.44), 1.5: mhz_to_ev(18.5)}
    base = h_with_hyperfine(1, 1.5, a, 0.1, NA_G_I, 0.001)
    assert np.allclose(base, h_with_hyperfine(1, 1.5, a, 0.1, NA_G_I, 0.001, b_quad_by_j_ev={0.5: big}), atol=1e-14)
    assert np.abs(quadrupole_operator(0.5, 1.5, np.eye(4), np.eye(4))).max() == 0
    assert np.abs(quadrupole_operator(1.5, 0.5, np.eye(4), np.eye(4))).max() == 0


@pytest.mark.implementation
def test_f_weights_pure_at_zero_field_and_mixed_in_a_field():
    h0 = h_na_3p(0.0, _ZETA)
    nudge = 1e-13 * np.kron(np.eye(6), angular_momentum_ops(1.5)[2])               # picks definite m_F inside each F level
    w, v = np.linalg.eigh(h0 + nudge)
    fw = f_weights(h0, 1, NA_I, v)
    assert fw.f_purity.min() > 1 - 1e-9
    p32 = fw[fw["p_j1.5"] > 0.5]
    assert sorted(p32.f.tolist()) == sorted([0.0] + [1.0] * 3 + [2.0] * 5 + [3.0] * 7)
    assert (fw.m_f.abs() <= fw.f + 1e-9).all()
    wb, vb = np.linalg.eigh(h_na_3p(0.3, _ZETA))
    assert f_weights(h_na_3p(0.3, _ZETA), 1, NA_I, vb).f_purity.min() < 0.9            # F stops being a label at 0.3 T


@pytest.mark.spectroscopic
def test_3p32_intervals_against_measured_values_not_used_in_the_model():
    """Das and Natarajan (arXiv:0711.0067): Delta nu(F'=3-2) = 3A+B = 58.310(11) MHz and Delta nu(F'=2-1) = 2A-B = 34.339(11) MHz.
    The model's A and B come from Yei et al. 1993, a different measurement. The F'=1-0 interval (15.810 MHz) is derived, not
    measured. A pass means agreement within 3 sigma (the model sits 1.5 and 0.5 sigma above); the dipole-only model misses by
    more than 100 sigma."""
    w = np.unique(np.round(_p32_levels(True), 5))
    d32, d21 = w[3] - w[2], w[2] - w[1]
    assert abs(d32 - 58.310) < 3 * 0.011 and abs(d21 - 34.339) < 3 * 0.011
    wo = np.unique(np.round(_p32_levels(False), 5))
    assert abs(wo[3] - wo[2] - 58.310) > 100 * 0.011
