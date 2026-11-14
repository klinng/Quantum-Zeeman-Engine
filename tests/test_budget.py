import numpy as np
import pytest

from src.budget import (CA_SCENARIO_SIGMA_CM1, _ca_admixture, _ca_admixture_constrained, budget_table, g_factor_rows, rows)


@pytest.fixture(scope="module")
def table():
    return budget_table(n=40)


def test_every_system_is_present(table):
    assert {"Sodium", "Hydrogen n=2", "Hydrogen H-alpha", "Calcium 4s4p"} <= set(table.system)


def test_every_row_declares_its_sigma_source(table):
    assert set(table.sigma_source) <= {"measured", "assumed"}
    assert (table.sigma >= 0).all()


def test_sodium_hyperfine_row_matches_exact_propagation(table):
    r = table[table.observable == "3P1/2 hyperfine splitting"].iloc[0]
    assert r.sigma == pytest.approx(0.26, rel=0.3)
    assert r.value == pytest.approx(188.88, abs=0.2)


def test_no_calcium_hyperfine_row():
    assert not any("hyperfine" in r.observable for r in rows() if r.system == "Calcium 4s4p")


def test_g_factor_rows_flag_the_free_electron_gap():
    g = g_factor_rows().set_index("observable")
    assert g.loc["g_J 3P1/2", "verdict"] == "consistent"
    assert g.loc["g_J 3P3/2", "verdict"] == "consistent"
    assert g.loc["g_J 3S1/2", "verdict"] == "disagreement"          # model uses the free-electron g_s
    assert abs(g.loc["g_J 3S1/2", "value"] - g.loc["g_J 3S1/2", "reference"]) < 5e-5


def _ca_levels():
    from src.two_electron import CA_LEVELS_CM1
    return [CA_LEVELS_CM1[k] for k in ("3P0", "3P1", "3P2", "1P1")]


def test_calcium_budget_uses_the_four_parameter_model():
    from src.two_electron import calcium_four_parameter_fit
    eps2 = calcium_four_parameter_fit(*_ca_levels())["eps2"]
    assert _ca_admixture(*_ca_levels()) == pytest.approx(eps2, rel=1e-14)
    assert eps2 == pytest.approx(6.134e-5, rel=1e-3)
    # the old three-parameter model (zeta_A tied to zeta_S) gives 7.8e-5: eps^2 scales as zeta_A^2, (52.68/46.75)^2 = 1.27
    old = _ca_admixture_constrained(*_ca_levels())
    assert old == pytest.approx(7.785e-5, rel=1e-3)
    assert old / eps2 == pytest.approx((52.6807 / 46.7530) ** 2, rel=5e-3)


def test_calcium_budget_row_is_a_scenario_and_matches_propagation(table):
    r = table[table.observable == "singlet admixture in 3P1"].iloc[0]
    assert CA_SCENARIO_SIGMA_CM1 == 0.001
    assert r.sigma_source == "assumed" and "SCENARIO" in r.note and "not a standard error" in r.note
    assert r.nominal == pytest.approx(6.134e-5, rel=1e-3)
    assert r.sigma == pytest.approx(1.477e-7, rel=0.3)           # n = 40 draws: ~11 % sampling error of sigma, so wide tolerance
    assert r.sigma > 1e-7                                          # the stale value was 1.3e-9


def test_table_keeps_std_mc_sem_and_nominal_apart(table):
    mc = table[table.n_samples > 0]
    assert np.allclose(mc.mc_sem, mc.sigma / np.sqrt(mc.n_samples))
    assert np.allclose(mc.sigma_rel_sampling, 1 / np.sqrt(2 * (mc.n_samples - 1)))
    assert (mc.mc_sem < mc.sigma).all()
    # 'value' is still the Monte Carlo mean; 'nominal' is the deterministic central calculation
    h = mc[mc.observable.str.contains("crossing")].iloc[0]
    from src.uncertainty import hydrogen_crossing_gauss
    assert h.nominal == pytest.approx(hydrogen_crossing_gauss(), rel=1e-14)
    assert abs(h.value - h.nominal) < 5 * h.mc_sem                  # the MC mean differs from nominal only by sampling noise
    g = table[table.observable.str.startswith("g_J")]
    assert (g.n_samples == 0).all() and g.mc_sem.isna().all() and (g.value == g.nominal).all()
