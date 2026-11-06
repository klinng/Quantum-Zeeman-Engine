r"""Two valence electrons (alkaline-earth-like atoms, e.g. calcium 4s4p): singlet and triplet P terms in a magnetic field.

One electron sits in an s orbital (l = 0) and one in a p orbital (l = 1). Basis: |m_l> x |m_s1> x |m_s2>, 12 states.

    H = E_t + DELTA (1/4 - s1.s2) + zeta l.s2 + mu_B B (L_z + g_s (S1z + S2z))

* DELTA (1/4 - s1.s2) is the exchange splitting: zero for the triplet (S = 1), DELTA for the singlet (S = 0).
* zeta l.s2 is the spin-orbit coupling of the p electron. Inside the triplet it gives the Lande interval rule
  (E(3P2) - E(3P1)) : (E(3P1) - E(3P0)) = 2 : 1, and it is NOT diagonal in S: it mixes 1P1 into 3P1. That mixing
  makes the "forbidden" intercombination line 1S0 - 3P1 (657 nm in calcium) weakly allowed.
* The Zeeman term is the same as for one electron, with the total spin S = s1 + s2.
Not modelled: spin-spin interaction (large in helium, which is why this model suits calcium rather than helium),
configuration mixing with other levels (4s3d), and hyperfine structure. Energies in eV, fields in tesla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .constants import G_S, MU_B_EV, ev_to_wavenumber, wavenumber_to_ev
from .hamiltonians import angular_momentum_ops

_LX, _LY, _LZ = angular_momentum_ops(1)
_SX, _SY, _SZ = angular_momentum_ops(0.5)
_I3, _I2 = np.eye(3), np.eye(2)


def _op(l=None, s1=None, s2=None) -> np.ndarray:
    """Operator on the 12-dimensional space from factors acting on (orbital, spin 1, spin 2)."""
    return np.kron(np.kron(_I3 if l is None else l, _I2 if s1 is None else s1), _I2 if s2 is None else s2)


def operators() -> dict[str, np.ndarray]:
    o = {}
    for c, lc, sc_ in zip("xyz", (_LX, _LY, _LZ), (_SX, _SY, _SZ)):
        o["L" + c] = _op(l=lc)
        o["s1" + c] = _op(s1=sc_)
        o["s2" + c] = _op(s2=sc_)
        o["S" + c] = o["s1" + c] + o["s2" + c]
    return o


def h_two_electron(b_tesla: float, zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0, g_s: float = G_S) -> np.ndarray:
    """12x12 Hamiltonian (eV). e_t_ev is the energy of the triplet term before spin-orbit coupling."""
    o = operators()
    s1s2 = sum(o["s1" + c] @ o["s2" + c] for c in "xyz")
    ls2 = sum(o["L" + c] @ o["s2" + c] for c in "xyz")
    h = e_t_ev * np.eye(12) + delta_ev * (0.25 * np.eye(12) - s1s2) + zeta_ev * ls2
    return h + MU_B_EV * b_tesla * (o["Lz"] + g_s * o["Sz"])


def classify_states(b_tesla: float, zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0, g_s: float = G_S) -> pd.DataFrame:
    """Eigenstates with their labels. A tiny extra J_z term is added at B = 0 only to pick definite M_J inside degenerate levels.

    Columns: energy_ev, mj (<J_z>), j (from <J^2>), singlet (probability of S = 0), label ('3P0','3P1','3P2','1P1').
    The label of a level is its dominant character: with 1P1 and 3P1 mixed by the spin-orbit term, the J = 1 level
    that is mostly singlet is called 1P1 and the other 3P1.
    """
    o = operators()
    jz = o["Lz"] + o["Sz"]
    j2 = sum((o["L" + c] + o["S" + c]) @ (o["L" + c] + o["S" + c]) for c in "xyz")
    p_singlet = 0.25 * np.eye(12) - sum(o["s1" + c] @ o["s2" + c] for c in "xyz")
    h = h_two_electron(b_tesla, zeta_ev, delta_ev, e_t_ev, g_s)
    if b_tesla == 0.0:
        h = h + 1e-10 * jz
    w, v = np.linalg.eigh(h)
    ex = lambda m, k: float(np.real(v[:, k].conj() @ m @ v[:, k]))
    rows = []
    for k in range(12):
        j2v = ex(j2, k)
        j = (-1 + np.sqrt(1 + 4 * j2v)) / 2
        rows.append((w[k], ex(jz, k), j, ex(p_singlet, k)))
    df = pd.DataFrame(rows, columns=["energy_ev", "mj", "j", "singlet"])
    df["j"] = df.j.round()
    df["label"] = np.where(df.j == 0, "3P0", np.where(df.j == 2, "3P2", np.where(df.singlet > 0.5, "1P1", "3P1")))
    return df


def zero_field_levels_cm1(zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0) -> dict[str, float]:
    """Energies (cm^-1) of 3P0, 3P1, 3P2 and 1P1 from diagonalisation at zero field."""
    df = classify_states(0.0, zeta_ev, delta_ev, e_t_ev)
    return {lab: float(ev_to_wavenumber(g.energy_ev.mean())) for lab, g in df.groupby("label")}


def fit_to_levels(e3p0_cm1: float, e3p1_cm1: float, e3p2_cm1: float, e1p1_cm1: float):
    """Fit (zeta, Delta, E_t) so the model reproduces the four term energies (3 parameters, 4 numbers: one consistency check).
    Returns (zeta_eV, delta_eV, e_t_eV, residuals_cm1 model minus input)."""
    target = np.array([e3p0_cm1, e3p1_cm1, e3p2_cm1, e1p1_cm1])

    def resid(p):
        lv = zero_field_levels_cm1(wavenumber_to_ev(p[0]), wavenumber_to_ev(p[1]), wavenumber_to_ev(p[2]))
        return np.array([lv["3P0"], lv["3P1"], lv["3P2"], lv["1P1"]]) - target

    p0 = [e3p2_cm1 - e3p1_cm1, e1p1_cm1 - e3p1_cm1, e3p1_cm1]
    sol = least_squares(resid, p0, x_scale=[100, 1000, 10000], xtol=1e-14, ftol=1e-14)
    z, d, et = (wavenumber_to_ev(x) for x in sol.x)
    return z, d, et, resid(sol.x)


def singlet_admixture_in_3p1(zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0) -> float:
    """Probability of S = 0 character in the 3P1 level (the squared mixing amplitude epsilon^2)."""
    df = classify_states(0.0, zeta_ev, delta_ev, e_t_ev)
    return float(df[df.label == "3P1"].singlet.mean())


def g_factor(label: str, zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0, b_tesla: float = 1e-4) -> float:
    """g_J of a level from the model's own eigenvalues at a tiny field: (E(M_J = +J) - E(M_J = -J)) / (2 J mu_B B)."""
    if label == "3P0":
        raise ValueError("3P0 (J = 0) has no Zeeman splitting")
    df = classify_states(b_tesla, zeta_ev, delta_ev, e_t_ev)
    g = df[df.label == label].sort_values("energy_ev")
    j = float(g.j.iloc[0])
    return float((g.energy_ev.iloc[-1] - g.energy_ev.iloc[0]) / (2 * j * MU_B_EV * b_tesla))


def intercombination_rate(eps2: float, a_singlet_per_s: float, lambda_singlet_nm: float, lambda_triplet_nm: float) -> float:
    """Rough decay rate (1/s) of 3P1 to the ground state: the singlet admixture eps^2 times the 1P1 rate, scaled by
    (lambda_1P1 / lambda_3P1)^3 because spontaneous emission goes as the cube of the frequency. Assumes the singlet and the
    admixed triplet share the same dipole matrix element: an order-of-magnitude estimate, not a precise prediction."""
    return eps2 * a_singlet_per_s * (lambda_singlet_nm / lambda_triplet_nm) ** 3


# Calcium 4s4p term energies (cm^-1).
# 3P1: the measured 1S0-3P1 frequency 455 986 240 494 135.8 (3.4) Hz (tf.nist.gov/general/pdf/2197.pdf) divided by c
# gives 15210.0638 cm^-1 (the NIST Handbook table lists 15210.063).
# 3P0, 3P2, 1P1: NIST Handbook of Basic Atomic Spectroscopic Data, "Energy Levels of Neutral Calcium (Ca I)", whose
# source code is SC85 (Sugar and Corliss 1985): 15157.901, 15315.943 and 23652.304 cm^-1.
CA_3P1_MEASURED_HZ = 455_986_240_494_135.8
CA_LEVELS_CM1 = {"3P0": 15157.901, "3P1": 15210.0638, "3P2": 15315.943, "1P1": 23652.304}
