import numpy as np
import pytest

from src.budget import budget_table, g_factor_rows, rows


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
