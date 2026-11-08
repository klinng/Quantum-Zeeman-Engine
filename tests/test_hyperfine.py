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
