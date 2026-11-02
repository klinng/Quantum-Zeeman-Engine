"""Numerical eigensolvers for the Zeeman Hamiltonians."""
from __future__ import annotations

import numpy as np

from .constants import G_S
from .hamiltonians import h_total


def solve(l: int, b_tesla: float, zeta_ev: float, **kw):
    """Diagonalise H at one field value. Returns (energies ascending [eV], eigenvectors)."""
    h = h_total(l, b_tesla, zeta_ev, **kw)
    return np.linalg.eigh(h)


def energy_vs_field(l: int, b_values, zeta_ev: float, g_s: float = G_S) -> np.ndarray:
    """Energies for every B. Shape (len(b_values), 2*(2l+1)), each row sorted ascending."""
    b_values = np.asarray(b_values, dtype=float)
    out = np.empty((b_values.size, 2 * (2 * l + 1)))
    for i, b in enumerate(b_values):
        out[i] = np.linalg.eigvalsh(h_total(l, b, zeta_ev, g_s=g_s))
    return out


def track_levels(l: int, b_values, zeta_ev: float, g_s: float = G_S) -> np.ndarray:
    """Like energy_vs_field, but follows each level continuously through the field sweep.

    Because H conserves J_z = m_l + m_s, levels with different J_z may cross but levels with the
    same J_z never do. Tracking by eigenvector overlap gives smooth physical curves.
    """
    b_values = np.asarray(b_values, dtype=float)
    n = 2 * (2 * l + 1)
    energies = np.empty((b_values.size, n))
    w, v = np.linalg.eigh(h_total(l, b_values[0], zeta_ev, g_s=g_s))
    energies[0] = w
    for i in range(1, b_values.size):
        w_new, v_new = np.linalg.eigh(h_total(l, b_values[i], zeta_ev, g_s=g_s))
        overlap = np.abs(v.conj().T @ v_new) ** 2           # (old, new)
        order = np.argmax(overlap, axis=1)
        if len(set(order)) < n:                              # ambiguous: fall back to sorted
            order = np.arange(n)
        energies[i] = w_new[order]
        v = v_new[:, order]
    return energies


def zeeman_transitions(e_upper_ev, e_lower_ev):
    """All |upper - lower| energy differences (eV) between two level sets."""
    return np.abs(np.subtract.outer(np.asarray(e_upper_ev), np.asarray(e_lower_ev))).ravel()
