r"""Level 4: electric-dipole selection rules and Zeeman line intensities.

The dipole operator is a rank-1 spherical tensor, so by the Wigner-Eckart theorem its matrix
elements between |l m> states are proportional to a Wigner 3-j symbol. That single fact gives:

    * the selection rules   l' = l +/- 1,  m' - m = q in {-1, 0, +1}
    * the polarisation      q = 0 -> pi light,  q = +/-1 -> sigma light
    * the relative line intensities (squares of the matrix elements)

Spin is not touched by the dipole operator, so the spin quantum number m_s is conserved. Mixing of
the (m_l, m_s) states by spin-orbit coupling is handled automatically: we diagonalise both
manifolds numerically and project the dipole operator onto the eigenvectors.
"""
from __future__ import annotations

from math import factorial, sqrt

import numpy as np
import pandas as pd

from .constants import MU_B_EV, ev_to_wavenumber
from .hamiltonians import G_S, h_total

POLARISATION = {-1: "sigma-", 0: "pi", 1: "sigma+"}


def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Wigner 3-j symbol for integer angular momenta (Racah formula)."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    delta = (factorial(j1 + j2 - j3) * factorial(j1 - j2 + j3) * factorial(-j1 + j2 + j3)
             / factorial(j1 + j2 + j3 + 1))
    pref = sqrt(delta * factorial(j1 + m1) * factorial(j1 - m1) * factorial(j2 + m2)
                * factorial(j2 - m2) * factorial(j3 + m3) * factorial(j3 - m3))
    total = 0.0
    k = 0
    while True:
        args = (k, j1 + j2 - j3 - k, j1 - m1 - k, j2 + m2 - k, j3 - j2 + m1 + k, j3 - j1 - m2 + k)
        if min(args) >= 0:
            term = 1.0
            for a in args:
                term /= factorial(a)
            total += (-1) ** k * term
        elif k > j1 + j2 + j3 + 1:
            break
        k += 1
        if k > 2 * (j1 + j2 + j3) + 2:
            break
    return (-1) ** (j1 - j2 - m3) * pref * total


def _c1(l_up: int, m_up: int, l_low: int, m_low: int, q: int) -> float:
    """Reduced orbital matrix element <l_up m_up| C^(1)_q |l_low m_low>."""
    if m_up - m_low != q:
        return 0.0
    return ((-1) ** m_up * sqrt((2 * l_up + 1) * (2 * l_low + 1))
            * wigner_3j(l_up, 1, l_low, 0, 0, 0) * wigner_3j(l_up, 1, l_low, -m_up, q, m_low))


def dipole_matrices(l_up: int, l_low: int) -> dict[int, np.ndarray]:
    """Dipole operator for each polarisation q, shape (dim_up, dim_low), in the |m_l, m_s> bases.

    Basis ordering matches hamiltonians.py: m_l descending (outer), m_s = +1/2, -1/2 (inner).
    """
    ml_up = list(range(l_up, -l_up - 1, -1))
    ml_low = list(range(l_low, -l_low - 1, -1))
    out = {}
    for q in (-1, 0, 1):
        d = np.zeros((2 * len(ml_up), 2 * len(ml_low)))
        for iu, mu in enumerate(ml_up):
            for il, ml in enumerate(ml_low):
                val = _c1(l_up, mu, l_low, ml, q)
                for s in (0, 1):                       # spin conserved
                    d[2 * iu + s, 2 * il + s] = val
        out[q] = d
    return out


def zeeman_lines(l_up: int, l_low: int, b_tesla: float, zeta_up_ev: float, zeta_low_ev: float = 0.0,
                 gap_ev: float = 0.0, g_s: float = G_S, normalize: bool = True,
                 threshold: float = 1e-10) -> pd.DataFrame:
    """All allowed Zeeman components between an upper and a lower manifold.

    Parameters
    ----------
    gap_ev : energy of the upper manifold's centre-of-gravity reference above the lower one
             (the H_0 term), so that line energies equal real transition energies.
    normalize : scale intensities so every upper state emits a total of 1 on average.

    Returns a DataFrame with columns: upper, lower (level indices, ascending energy), q,
    polarisation, energy_ev, wavenumber_cm1, intensity.
    """
    eu, vu = np.linalg.eigh(h_total(l_up, b_tesla, zeta_up_ev, e0_ev=gap_ev, g_s=g_s))
    el, vl = np.linalg.eigh(h_total(l_low, b_tesla, zeta_low_ev, g_s=g_s))
    dmat = dipole_matrices(l_up, l_low)
    amps = {q: vu.conj().T @ d @ vl for q, d in dmat.items()}      # (n_up, n_low)
    total = sum((np.abs(a) ** 2).sum() for a in amps.values())
    scale = (len(eu) / total) if normalize and total > 0 else 1.0
    rows = []
    for q, a in amps.items():
        inten = np.abs(a) ** 2 * scale
        for i in range(len(eu)):
            for j in range(len(el)):
                if inten[i, j] > threshold:
                    e = eu[i] - el[j]
                    rows.append((i, j, q, POLARISATION[q], e, ev_to_wavenumber(e), inten[i, j]))
    cols = ["upper", "lower", "q", "polarisation", "energy_ev", "wavenumber_cm1", "intensity"]
    return pd.DataFrame(rows, columns=cols).sort_values("energy_ev").reset_index(drop=True)


def shift_in_bohr_units(df: pd.DataFrame, centre_ev: float, b_tesla: float) -> np.ndarray:
    """Line shifts from a reference energy in units of mu_B * B (handy for textbook patterns)."""
    return (df["energy_ev"].to_numpy() - centre_ev) / (MU_B_EV * b_tesla)
