r"""Helium n = 2 inputs and the calibration of the helium model (src/helium.py).

Sources (both read from the pages, not from memory):
* NIST Handbook of Basic Atomic Spectroscopic Data, "Energy Levels of Neutral Helium (He I)", reference code M02
  (physics.nist.gov/PhysRefData/Handbook/Tables/heliumtable5.htm).
* K. Kato, T. D. G. Skinner and E. A. Hessels, arXiv:1807.07923: 2^3P_2 - 2^3P_1 = 2 291 176 590(25) Hz (measured).

The model has six parameters (E_dir 1s2s, G^0, E_dir 1s2p, G^1, zeta_S, rho). Seven numbers enter the procedure: the six NIST
levels (3S1, 1S0, 3P2, 3P1, 3P0, 1P1) and the Kato interval. They are NOT a joint least-squares problem: the procedure is a
sequential exact solve. Kato's interval REPLACES the table's 3P2-3P1 difference in the 2x2 solve for (zeta_S, rho), together
with the NIST difference 3P0-3P1; the NIST 3P2 and 3P1 absolute levels still enter the 3P centroid. So there are six independent
constraints for six parameters (zero degrees of freedom, no chi^2, no objective function), plus one redundant observation: the
table's 3P2-3P1 interval, which is not fitted but compared with Kato (they agree to 3e-5 cm^-1, i.e. within the table's 1e-4
rounding). Fitting them is therefore a CALIBRATION, not a test of the model; it makes the model a parametrised description of helium that can then PREDICT things
that were not fitted (Zeeman patterns, g-factors, level mixing). Spin-orbit/spin-other-orbit and spin-spin cannot be separated
here: zeta_S and rho are effective parameters. zeta_A (singlet-triplet spin-orbit mixing) is set to zero.
"""
from __future__ import annotations

from .constants import wavenumber_to_ev
from .helium import fit_triplet_p_intervals

C_CM_PER_S = 29_979_245_800.0                       # c in cm/s, so 1 cm^-1 = 29979.2458 MHz

HE_LEVELS_CM1 = {"3S1": 159855.9745, "1S0": 166277.4403, "3P2": 169086.7666, "3P1": 169086.8430, "3P0": 169087.8309,
                 "1P1": 171134.8970}
KATO_3P2_3P1_MHZ = 2291.176590                      # measured, uncertainty 25 Hz


def mhz_to_cm1(mhz: float) -> float:
    return mhz * 1e6 / C_CM_PER_S


def helium_parameters(levels=HE_LEVELS_CM1, e12_mhz: float = KATO_3P2_3P1_MHZ) -> dict[str, float]:
    """Calibrate (G^0, G^1, E_dir for 1s2s and 1s2p, zeta_S, rho) in cm^-1 from the six levels and the precise 2^3P interval.

    G^0 = (E(2^1S) - E(2^3S)) / 2; G^1 = (E(2^1P_1) - centroid(2^3P)) / 2, centroid = sum (2J+1) E_J / 9, because the
    spin-orbit and spin-spin terms both average to zero over the triplet. zeta_S and rho come from E0 - E1 (NIST) and
    E1 - E2 (the Kato et al. measurement, which is 4 digits more precise than the table)."""
    lv = levels
    g0 = (lv["1S0"] - lv["3S1"]) / 2
    centroid = (5 * lv["3P2"] + 3 * lv["3P1"] + lv["3P0"]) / 9
    g1 = (lv["1P1"] - centroid) / 2
    e01 = lv["3P0"] - lv["3P1"]
    zeta, rho = fit_triplet_p_intervals(e01, mhz_to_cm1(e12_mhz))
    return {"G0": g0, "G1": g1, "E_dir_s": (lv["1S0"] + lv["3S1"]) / 2, "E_dir_p": (lv["1P1"] + centroid) / 2,
            "zeta_S": zeta, "rho": rho, "centroid_3P": centroid}


def helium_parameters_ev(**kw) -> dict[str, float]:
    return {k: float(wavenumber_to_ev(v)) for k, v in helium_parameters(**kw).items()}


def fig_helium_2p_fan(b_max: float = 3.0, n: int = 121):
    """Zeeman fan of the 12 levels of He 1s2p with the calibrated parameters (a PREDICTION: nothing here is fitted to a field).
    Left: energies relative to the 2^3P_2 level; right: the same at low field, where J is still a good quantum number."""
    import numpy as np
    import matplotlib.pyplot as plt
    from .constants import ev_to_wavenumber
    from .helium import h_helium_1s2p
    q = helium_parameters_ev()
    bs = np.linspace(1e-4, b_max, n)
    ev = np.array([np.linalg.eigvalsh(h_helium_1s2p(b, q["G1"], q["zeta_S"], q["rho"], 0.0, q["E_dir_p"])) for b in bs])
    base = HE_LEVELS_CM1["3P2"]
    rel = ev_to_wavenumber(ev) - base
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    triplet = rel[:, :9]
    for k in range(9):
        ax1.plot(bs, triplet[:, k], color="tab:blue", lw=1.2)
    ax1.set_xlabel("magnetic field B (T)")
    ax1.set_ylabel(r"energy above $2^3P_2$ at B = 0 (cm$^{-1}$)")
    ax1.set_title(r"He $2^3P$ (nine levels), calibrated to NIST and Kato et al.", fontsize=10)
    bl = np.linspace(1e-4, 0.3, 301)
    low = np.array([ev_to_wavenumber(np.linalg.eigvalsh(h_helium_1s2p(b, q["G1"], q["zeta_S"], q["rho"], 0.0, q["E_dir_p"])))[:9]
                    for b in bl]) - base
    for k in range(9):
        ax2.plot(bl, low[:, k], color="tab:blue", lw=1.2)
    ax2.set_xlabel("magnetic field B (T)")
    ax2.set_ylabel(r"energy above $2^3P_2$ at B = 0 (cm$^{-1}$)")
    ax2.set_title("Low field: the J = 0 level lies 0.99 cm$^{-1}$ above J = 1", fontsize=10)
    fig.tight_layout()
    return fig
