"""Tests for src/helium_data.py: calibration of the helium model to NIST levels and the Kato et al. interval."""
import numpy as np
import pytest

from src.constants import ev_to_wavenumber as cm
from src.helium import (direct_f_hartree, exchange_g_hartree, h_helium_1s2p, h_helium_1s2s)
from src.helium_data import (HE_LEVELS_CM1, KATO_3P2_3P1_MHZ, helium_parameters, helium_parameters_ev, mhz_to_cm1)


def test_two_sources_agree_on_the_3p2_3p1_interval():
    nist = HE_LEVELS_CM1["3P1"] - HE_LEVELS_CM1["3P2"]                   # tabulated to 1e-4 cm^-1
    assert abs(nist - mhz_to_cm1(KATO_3P2_3P1_MHZ)) < 1.5e-4


def test_calibrated_1s2p_reproduces_the_nist_levels():
    q = helium_parameters_ev()
    w = cm(np.linalg.eigvalsh(h_helium_1s2p(0.0, q["G1"], q["zeta_S"], q["rho"], 0.0, q["E_dir_p"])))
    lv = HE_LEVELS_CM1
    expected = sorted([lv["3P2"]] * 5 + [lv["3P1"]] * 3 + [lv["3P0"]] + [lv["1P1"]] * 3)
    # the 3P2 level moves by 2.5e-5 cm^-1 because the precise Kato interval replaces the table difference
    assert np.allclose(w, expected, atol=4e-5)


def test_calibrated_1s2s_reproduces_the_nist_levels():
    q = helium_parameters_ev()
    w = cm(np.linalg.eigvalsh(h_helium_1s2s(0.0, q["G0"], q["E_dir_s"])))
    assert np.allclose(w, sorted([HE_LEVELS_CM1["3S1"]] * 3 + [HE_LEVELS_CM1["1S0"]]), atol=1e-6)


def test_exchange_values():
    p = helium_parameters()
    assert p["G0"] == pytest.approx(3210.7329, abs=1e-4) and p["G1"] == pytest.approx(1023.99, abs=0.01)


def test_triplet_p_is_inverted_and_breaks_lande_rule():
    p = helium_parameters()
    assert p["zeta_S"] < 0 and p["rho"] > 0
    lv = HE_LEVELS_CM1
    ratio = (lv["3P2"] - lv["3P1"]) / (lv["3P1"] - lv["3P0"])            # Lande: 2
    assert abs(ratio) < 0.1


def test_crude_hydrogenic_exchange_overestimates_real_helium():
    """The hydrogenic orbitals (Z = 2 and 1) give G about 3.7 times (G^0) and 2.5 times (G^1) too large: they are only an order of magnitude."""
    p = helium_parameters()
    g0 = 2 * exchange_g_hartree(0, (2, 0)) * 219474.6313 / 2
    g1 = 2 * exchange_g_hartree(1, (2, 1)) * 219474.6313 / 2
    assert 3 < g0 / p["G0"] < 4.5 and 2 < g1 / p["G1"] < 3
