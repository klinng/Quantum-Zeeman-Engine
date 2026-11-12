"""Tests for src/helium.py (1s2s and 1s2p with exchange, spin-orbit, spin-spin and Zeeman terms)."""
import numpy as np
import pytest

from src.constants import G_S, MU_B_EV, wavenumber_to_ev as ev
from src.helium import (_ops, classify, classify_tracked, direct_f_hartree, exchange_g_hartree, fit_triplet_p_intervals, h_exchange,
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


# --- labelling at finite field and the omitted zeta_A term (helium audit) -------------------------------------

def _he_h(b, za=0.0):
    from src.helium_data import helium_parameters_ev
    q = helium_parameters_ev()
    return h_helium_1s2p(b, q["G1"], q["zeta_S"], q["rho"], za, q["E_dir_p"])


def test_classify_j_weights_sum_to_one_and_are_pure_at_zero_field():
    lv = classify(0.0, _he_h(0.0))
    assert np.allclose(lv.p_j0 + lv.p_j1 + lv.p_j2, 1.0)
    assert lv.j_purity.min() > 1 - 1e-6
    assert sorted(lv.j) == [0] + [1] * 6 + [2] * 5          # 2^3P_0, (2^3P_1 + 2^1P_1), 2^3P_2


def test_rounded_j_label_would_mislead_at_finite_field():
    """The reason classify no longer rounds <J^2>: at 1 T some states are ~50/50 mixtures of J = 1 and 2."""
    lv = classify(1.0, _he_h(1.0))
    assert lv.j_purity.min() < 0.55
    frac = (lv.j2 - lv.j2.round()).abs()
    assert frac.max() > 0.3                                  # <J^2> far from any j(j+1) integer
    assert np.allclose(lv.m_j - lv.mj, 0, atol=1e-6)         # M_J itself stays exact


def test_m_pm2_states_stay_pure_j2_at_all_fields():
    for b in (0.1, 1.0, 3.0):
        lv = classify(b, _he_h(b))
        edge = lv[lv.m_j.abs() == 2]
        assert len(edge) == 2 and (edge.j_purity > 1 - 1e-9).all() and (edge.j == 2).all()


def test_tracked_parentage_labels_and_conserved_m():
    bs = np.concatenate([[1e-4], np.linspace(0.05, 3.0, 40)])
    df = classify_tracked(bs, _he_h)
    first = df[df.b_tesla == bs[0]]
    assert first.parent.value_counts().to_dict() == {"2^3P_2": 5, "2^3P_1": 3, "2^3P_0": 1, "2^1P_1": 3}
    for _, grp in df.groupby("level"):
        assert grp.m_j.nunique() == 1                        # M_J is conserved along every tracked level
    sing = df[df.parent == "2^1P_1"]
    assert sing.singlet.min() > 0.999999                     # the singlet parent stays singlet to 3 T


def test_zeta_a_selection_rules_and_matrix_element():
    o, _, _, ldiff, _ = _ops()
    jz = o["Lz"] + o["Sz"]
    j2 = sum((o["L" + c] + o["S" + c]) @ (o["L" + c] + o["S" + c]) for c in "xyz")
    assert np.abs(ldiff @ jz - jz @ ldiff).max() < 1e-12     # M_J conserved
    assert np.abs(ldiff @ j2 - j2 @ ldiff).max() < 1e-12     # J conserved at B = 0
    # in the J = 1 sector the 3P1-1P1 element is sqrt(2) in magnitude
    lv, v = np.linalg.eigh(_he_h(0.0) + 1e-3 * (o["Lz"] + o["Sz"]))
    a = v.conj().T @ ldiff @ v
    big = np.abs(a)[np.abs(a) > 1e-9]
    assert np.allclose(sorted(set(np.round(big, 6))), [round(2 ** 0.5, 6)])
    # the Zeeman operator has no S = 0 <-> S = 1 element
    zee = o["Lz"] + 2.0023 * o["Sz"]
    s1s2 = _ops()[4]
    psing = 0.25 * np.eye(12) - s1s2                          # singlet projector
    assert np.abs(psing @ zee @ (np.eye(12) - psing)).max() < 1e-12


def test_singlet_admixture_of_triplet_p1_is_second_order_in_zeta_a():
    """P(S=0 in 2^3P_1) = (sqrt2 zeta_A / dE)^2 with dE the 2^1P_1 - 2^3P_1 gap (here 2048.13 cm^-1); and the
    omitted term shifts 2^1P_1 / 2^3P_1 by +- V^2/dE (4.755 MHz for zeta_A = 0.403 cm^-1)."""
    from src.constants import wavenumber_to_ev, ev_to_wavenumber
    za_cm = 0.40
    lv0 = classify(0.0, _he_h(0.0)).sort_values("energy_ev")
    lv1 = classify(0.0, _he_h(0.0, wavenumber_to_ev(za_cm))).sort_values("energy_ev")
    gap = ev_to_wavenumber(lv0.energy_ev.iloc[-1] - lv0.energy_ev.iloc[5])
    eps2 = (2 ** 0.5 * za_cm / gap) ** 2
    t1 = lv1[(lv1.j == 1) & (lv1.singlet < 0.5)]
    assert t1.singlet.mean() == pytest.approx(eps2, rel=0.02)
    shift = ev_to_wavenumber(lv1.energy_ev.iloc[9] - lv0.energy_ev.iloc[9])          # singlet level moves up
    assert shift == pytest.approx(eps2 * gap, rel=0.02)
