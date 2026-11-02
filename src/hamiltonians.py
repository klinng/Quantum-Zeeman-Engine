r"""Matrix builders for a single electron with orbital momentum l and spin 1/2.

Basis: |l, m_l> (x) |s, m_s>, ordered with m_l descending (outer) and m_s = +1/2, -1/2 (inner).
Energies are in eV, magnetic field in tesla (hbar = 1 inside the operators).

    H = H_0 + H_SO + H_Z
    H_SO = zeta * L.S
    H_Z  = mu_B * B * (g_l L_z + g_s S_z)
"""
from __future__ import annotations

import numpy as np

from .constants import G_L, G_S, MU_B_EV


def angular_momentum_ops(j: float):
    """Return (Jx, Jy, Jz) for angular momentum j in the |j, m> basis, m descending (hbar = 1)."""
    dim = int(round(2 * j + 1))
    m = j - np.arange(dim)
    jz = np.diag(m).astype(complex)
    jp = np.zeros((dim, dim), dtype=complex)          # raising operator
    for k in range(1, dim):
        jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    jm = jp.conj().T
    jx = (jp + jm) / 2
    jy = (jp - jm) / (2j)
    return jx, jy, jz


def coupled_ops(l: int):
    """Operators L and S on the product space. Returns dict with Lx..Sz and the dimension."""
    lx, ly, lz = angular_momentum_ops(l)
    sx, sy, sz = angular_momentum_ops(0.5)
    il, is_ = np.eye(2 * l + 1), np.eye(2)
    ops = {
        "Lx": np.kron(lx, is_), "Ly": np.kron(ly, is_), "Lz": np.kron(lz, is_),
        "Sx": np.kron(il, sx), "Sy": np.kron(il, sy), "Sz": np.kron(il, sz),
    }
    return ops


def h_spin_orbit(l: int, zeta_ev: float) -> np.ndarray:
    """H_SO = zeta * L.S"""
    o = coupled_ops(l)
    ls = o["Lx"] @ o["Sx"] + o["Ly"] @ o["Sy"] + o["Lz"] @ o["Sz"]
    return zeta_ev * ls


def h_zeeman(l: int, b_tesla: float, g_l: float = G_L, g_s: float = G_S) -> np.ndarray:
    """H_Z = mu_B B (g_l L_z + g_s S_z)"""
    o = coupled_ops(l)
    return MU_B_EV * b_tesla * (g_l * o["Lz"] + g_s * o["Sz"])


def h_total(l: int, b_tesla: float, zeta_ev: float, e0_ev: float = 0.0,
            g_l: float = G_L, g_s: float = G_S) -> np.ndarray:
    """Full Hamiltonian H_0 + H_SO + H_Z (H_0 is just the constant e0)."""
    dim = 2 * (2 * l + 1)
    return e0_ev * np.eye(dim, dtype=complex) + h_spin_orbit(l, zeta_ev) + h_zeeman(l, b_tesla, g_l, g_s)


def landé_g(l: int, j: float, s: float = 0.5, g_s: float = G_S) -> float:
    """Landé g-factor g_J for LS coupling (g_l = 1)."""
    if j == 0:
        return 0.0
    return 1 + (g_s - 1) * (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))
