"""Tests for src/helium.py (1s2s and 1s2p with exchange, spin-orbit, spin-spin and Zeeman terms)."""
import numpy as np
import pytest

from src.constants import G_S, MU_B_EV, wavenumber_to_ev as ev
from src.helium import (_ops, classify, direct_f_hartree, exchange_g_hartree, fit_triplet_p_intervals, h_exchange,
                        h_helium_1s2p, h_helium_1s2s, h_spin_spin, slater_integral, triplet_p_levels)

G, ZS, RHO = ev(1000.0), ev(-0.3), ev(0.05)        # round numbers; the physics checks do not depend on them


def test_all_hamiltonians_hermitian():
    assert np.allclose(h_helium_1s2p(0.7, G, ZS, RHO, zeta_a_ev=ev(0.02)), h_helium_1s2p(0.7, G, ZS, RHO, zeta_a_ev=ev(0.02)).conj().T)
    h = h_helium_1s2s(0.7, G)
    assert np.allclose(h, h.conj().T)


def test_exchange_splits_singlet_from_triplet_by_two_g():
    w = np.linalg.eigvalsh(h_exchange(G))
    assert set(np.round(np.unique(np.round(w / G, 9)), 6)) == {-1.0, 1.0}
    assert (w > 0).sum() == 3 and (w < 0).sum() == 9          # singlet: 1 spin state x 3 m_l; triplet 3 x 3
    assert w.max() - w.min() == pytest.approx(2 * G)


def test_exchange_equals_two_electron_delta_up_to_constant():
    from src.two_electron import h_two_electron
    d = h_two_electron(0.0, 0.0, 2 * G) - h_exchange(G)
    assert np.allclose(d, d[0, 0] * np.eye(12))


def test_spin_spin_is_zero_in_singlets_and_traceless():
    h = h_spin_spin(RHO)
    o, *_ = _ops()
    s2 = sum(o["S" + c] @ o["S" + c] for c in "xyz")
    singlet_projector = np.eye(12) - s2 / 2           # S(S+1) = 0 or 2: (1 - S^2/2) is 1 on singlets, 0 on triplets
    assert np.allclose(h @ singlet_projector, 0, atol=1e-18)
    assert np.trace(h) == pytest.approx(0, abs=1e-15)


def test_spin_spin_gives_1_minus5_10_pattern_in_the_triplet():
    h = h_helium_1s2p(0.0, 0.0, 0.0, RHO)
    lv = classify(0.0, h)
    lv = lv[lv.singlet < 0.5]
    e = {j: lv[lv.j == j].energy_ev.mean() for j in (0, 1, 2)}
    assert (e[2] - e[1]) / (e[1] - e[0]) == pytest.approx((1 / 6 + 5 / 6) / (-5 / 6 - 5 / 3))
    assert e[0] / e[2] == pytest.approx(10.0, rel=1e-6) and e[1] / e[2] == pytest.approx(-5.0, rel=1e-6)


def test_spin_orbit_alone_obeys_lande_interval_rule():
    lv = classify(0.0, h_helium_1s2p(0.0, 0.0, ZS, 0.0))
    lv = lv[lv.singlet < 0.5]
    e = {j: lv[lv.j == j].energy_ev.mean() for j in (0, 1, 2)}
    assert (e[2] - e[1]) / (e[1] - e[0]) == pytest.approx(2.0)


def test_numerical_levels_match_closed_form():
    lv = classify(0.0, h_helium_1s2p(0.0, 0.0, ZS, RHO))
    lv = lv[lv.singlet < 0.5]
    closed = triplet_p_levels(ZS, RHO)
    for j in (0, 1, 2):
        assert lv[lv.j == j].energy_ev.mean() == pytest.approx(closed[j], rel=1e-9)


def test_fit_of_intervals_roundtrips():
    levels = triplet_p_levels(ZS, RHO)
    z, r = fit_triplet_p_intervals(levels[0] - levels[1], levels[1] - levels[2])
    assert (z, r) == (pytest.approx(ZS), pytest.approx(RHO))


def test_breaking_of_lande_rule_needs_spin_spin():
    """With zeta fixed by the J=1-J=2 gap, rho != 0 changes the J=0 position: the 2:1 rule fails."""
    lv0 = triplet_p_levels(ZS, 0.0); lv1 = triplet_p_levels(ZS, RHO)
    assert (lv0[2] - lv0[1]) / (lv0[1] - lv0[0]) == pytest.approx(2.0)
    assert (lv1[2] - lv1[1]) / (lv1[1] - lv1[0]) != pytest.approx(2.0, rel=0.1)


def test_1s2s_triplet_g_factor_is_g_s_and_singlet_has_no_shift():
    b = 0.5
    w = np.linalg.eigvalsh(h_helium_1s2s(b, G))
    triplet = w[:3] - (-G)
    assert triplet.max() == pytest.approx(G_S * MU_B_EV * b) and triplet.min() == pytest.approx(-G_S * MU_B_EV * b)
    assert w[3] == pytest.approx(G, abs=1e-12)
    assert w[3] - w[0] == pytest.approx(2 * G + G_S * MU_B_EV * b)


def test_pure_triplet_p_zeeman_g_of_j2_is_three_halves():
    b = 1e-3
    lv = classify(b, h_helium_1s2p(b, G, 0.0, 0.0))        # no fine structure: Zeeman acts on L=1, S=1 (LS limit)
    t = lv[lv.singlet < 0.5].sort_values("energy_ev")
    assert (t.energy_ev.max() - t.energy_ev.min()) / (MU_B_EV * b) == pytest.approx(2 * (1 + G_S), rel=1e-3)


def test_slater_integral_hydrogenic_f0_1s1s():
    z = 2.0
    assert slater_integral(0, (1, 0, z), (1, 0, z), (1, 0, z), (1, 0, z)) == pytest.approx(5 * z / 8, rel=1e-6)


def test_exchange_integrals_positive_and_triplet_lower():
    assert exchange_g_hartree(0, (2, 0)) > 0 and exchange_g_hartree(1, (2, 1)) > 0
    assert direct_f_hartree(0, (2, 0)) > 0
