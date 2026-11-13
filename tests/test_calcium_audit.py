"""Calcium 4s4p audit tests. Three kinds, marked so they can be run and read separately:

  implementation  code agrees with algebra / an independent calculation
  calibration     a fit reproduces the numbers it was fitted to (closure only)
  spectroscopic   comparison with data that was NOT used in the fit: these decide whether the model is physically right.
                  Two of them LOCK IN KNOWN MISFITS (they pass because the discrepancy is documented), so a change in
                  the model that removes the misfit will make them fail on purpose and force the documentation to be updated.
Run only the data comparisons with:  python -m pytest -m spectroscopic
"""
import numpy as np
import pytest

from src.constants import ev_to_wavenumber as cm, wavenumber_to_ev as ev
from src.two_electron import (CA_LEVELS_CM1, CA_ODD_J1_LEVELS_CM1, calcium_fit_sensitivity, calcium_four_parameter_fit,
                              classify_states, displacement_for_rate, fit_to_levels, h_two_electron,
                              h_two_electron_general, intercombination_prediction, odd_j1_perturber_table,
                              singlet_admixture_in_3p1)

L = CA_LEVELS_CM1
FIT = calcium_four_parameter_fit(L["3P0"], L["3P1"], L["3P2"], L["1P1"])
KEYS = ("3P0", "3P1", "3P2", "1P1")


# ---------------------------------------------------------------- implementation

@pytest.mark.implementation
def test_general_hamiltonian_contains_the_one_parameter_model():
    h1 = h_two_electron(0.3, ev(80.0), ev(8000.0), ev(1.0))
    h2 = h_two_electron_general(0.3, ev(40.0), ev(40.0), ev(8000.0), ev(1.0))      # zeta_S = zeta_A = zeta / 2
    assert np.allclose(h1, h2, atol=1e-15)


@pytest.mark.implementation
def test_only_zeta_a_connects_singlet_and_triplet():
    lv = classify_states(0.0, ev(1e-6), ev(8000.0))                                  # no spin-orbit: pure terms
    assert lv.singlet.round(6).isin([0.0, 1.0]).all()
    h = h_two_electron_general(0.0, ev(50.0), 0.0, ev(8000.0)) + 1e-10 * np.diag(np.arange(12.0))
    w, v = np.linalg.eigh(h)
    sing = np.array([np.real(v[:, k].conj() @ (0.25 * np.eye(12) - sum(o @ o2 for o, o2 in _s1s2())) @ v[:, k]) for k in range(12)])
    assert np.minimum(sing, 1 - sing).max() < 1e-8                                   # zeta_S alone never mixes S


def _s1s2():
    from src.two_electron import operators
    o = operators()
    return [(o["s1" + c], o["s2" + c]) for c in "xyz"]


@pytest.mark.implementation
def test_zeta_s_from_the_3p0_3p2_interval_and_eps2_closed_form():
    assert FIT["zeta_S"] == pytest.approx((L["3P2"] - L["3P0"]) / 3.0, rel=1e-14)     # E(3P2) - E(3P0) = 3 zeta_S exactly
    h = h_two_electron_general(0.0, ev(FIT["zeta_S"]), ev(FIT["zeta_A"]), ev(FIT["Delta"]), ev(FIT["E_t"]))
    w, v = np.linalg.eigh(h + 1e-9 * np.diag(np.arange(12.0)))
    from src.two_electron import operators
    o = operators()
    psing = 0.25 * np.eye(12) - sum(o["s1" + c] @ o["s2" + c] for c in "xyz")
    j2 = sum((o["L" + c] + o["S" + c]) @ (o["L" + c] + o["S" + c]) for c in "xyz")
    k3p1 = [k for k in range(12) if abs(cm(w[k]) - L["3P1"]) < 1e-3 and abs(np.real(v[:, k].conj() @ j2 @ v[:, k]) - 2) < 1e-3]
    eps2 = np.mean([np.real(v[:, k].conj() @ psing @ v[:, k]) for k in k3p1])
    assert len(k3p1) == 3 and eps2 == pytest.approx(FIT["eps2"], rel=1e-5)             # eps^2 = (a - l1) / (l2 - l1)


@pytest.mark.implementation
def test_j_is_reported_with_its_purity():
    for b in (0.0, 1.0, 3.0):
        lv = classify_states(b, ev(105.3), ev(8388.0), ev(15263.0))
        assert np.allclose(lv.p_j0 + lv.p_j1 + lv.p_j2, 1.0)
        assert (lv.m_j - lv.mj).abs().max() < 1e-4
    assert classify_states(0.0, ev(105.3), ev(8388.0)).j_purity.min() > 1 - 1e-9
    assert classify_states(3.0, ev(105.3), ev(8388.0)).j_purity.min() > 1 - 2e-3       # fine structure >> Zeeman for calcium
    assert classify_states(10.0, ev(105.3), ev(8388.0)).j_purity.min() < 1 - 1e-3      # but not forever: the label degrades


@pytest.mark.implementation
def test_perturber_table_matches_nist_and_excludes_even_parity_and_j2():
    t = odd_j1_perturber_table().set_index("level")
    assert set(t.index) == {"4s5p 3P1", "4s5p 1P1", "3d4p 3D1"}                       # 4s3d (even) and 3d4p 3F2, 1D2 (J = 2) are absent
    assert t.loc["4s5p 3P1", "energy_cm1"] == 36554.749 and t.loc["3d4p 3D1", "energy_cm1"] == 38192.392
    assert (t.above_1P1_cm1 > 12000).all() and (t.above_3P1_cm1 > 21000).all()       # far compared with the 8442 cm^-1 of 1P1


@pytest.mark.implementation
def test_sensitivity_is_linear_in_the_assumed_level_uncertainty():
    a = calcium_fit_sensitivity(L, 0.01)
    b = calcium_fit_sensitivity(L, 0.02)
    assert np.allclose(b.std_err, 2 * a.std_err, rtol=1e-6)
    assert abs(a.attrs["correlation"].loc["zeta_A", "eps2"]) > 0.99                    # eps^2 and zeta_A carry the same information
    assert a.std_err.eps2 / a.value.eps2 == pytest.approx(0.024, rel=0.1)             # 2.4 % at sigma = 0.01 cm^-1 per level (scenario)


# ---------------------------------------------------------------- calibration (closure only)

@pytest.mark.calibration
def test_four_parameter_fit_closes_on_its_own_four_levels():
    h = h_two_electron_general(0.0, ev(FIT["zeta_S"]), ev(FIT["zeta_A"]), ev(FIT["Delta"]), ev(FIT["E_t"]))
    w = cm(np.linalg.eigvalsh(h))
    got = [w[0], w[1], w[4], w[9]]                                            # 3P0, 3P1, 3P2, 1P1 (1+3+5+3 states)
    assert np.allclose(got, [L[k] for k in KEYS], atol=1e-6)                           # four numbers, four parameters: no check left


# ---------------------------------------------------------------- spectroscopic comparisons

@pytest.mark.spectroscopic
def test_known_limitation_one_parameter_model_misses_the_measured_3p1_energy():
    """The 3-parameter model (zeta_S = zeta_A) has one check left. It FAILS it: residuals (model - NIST) of +0.060, -0.089 and
    +0.029 cm^-1 for 3P0, 3P1, 3P2. The 3P1 energy is known to hertz precision (the measured 455 986 240 494 135.8 Hz line),
    so a 0.09 cm^-1 (2.7 GHz) miss is a model error, not a data error. The earlier tolerance of 0.15 cm^-1 hid this."""
    *_, res = fit_to_levels(*(L[k] for k in KEYS))
    assert res[1] == pytest.approx(-0.0894, abs=2e-3)
    assert 0.08 < abs(res[1]) < 0.10 and abs(res[1]) > 100 * 1e-4 * 8                 # far above the 3P1 data precision (< 1e-6 cm^-1)


@pytest.mark.spectroscopic
def test_known_limitation_intercombination_lifetime_from_energies_alone():
    """Prediction (lifetime NOT used in the fit) against the two measured rates quoted in the report: 410 Hz and 375 Hz natural
    linewidths, A = 2 pi x width = 2576 and 2356 1/s. The model predicts 3589 1/s (0.279 ms): 1.4 to 1.5 times too fast.
    The one-parameter model gave 1.8 to 1.9. It is not explained by the data uncertainty: it would need the 3P0 / 3P2 levels
    wrong by ~0.2 cm^-1 (calcium_fit_sensitivity, scenario)."""
    pred = intercombination_prediction(L)
    for width_hz, ratio in ((410.0, 1.393), (375.0, 1.523)):
        assert pred["rate_per_s"] / (2 * np.pi * width_hz) == pytest.approx(ratio, rel=2e-3)
    assert 0.25 < pred["lifetime_ms"] < 0.30


@pytest.mark.spectroscopic
def test_displacement_diagnostic_the_1p1_mixing_cannot_supply_all_of_the_3p1_shift():
    """3P1 sits 0.518 cm^-1 below its interval-rule position. A 1P1 admixture that gives the MEASURED rate would displace it by
    only 0.34 to 0.37 cm^-1; the remaining 0.15 to 0.18 cm^-1 has to come from something else (other perturbers, spin-spin,
    relativistic and core effects). This identifies the size of what is missing; it does not identify the cause."""
    needed = [displacement_for_rate(2 * np.pi * w, L) for w in (410.0, 375.0)]
    assert needed[0] == pytest.approx(0.372, abs=0.003) and needed[1] == pytest.approx(0.340, abs=0.003)
    assert FIT["displacement_3P1"] == pytest.approx(0.5179, abs=1e-3)
    assert all(n < FIT["displacement_3P1"] for n in needed)
