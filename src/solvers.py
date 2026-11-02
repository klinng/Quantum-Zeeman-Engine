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


def track_from_builder(build, b_values, return_vectors: bool = False):
    """Follow every energy level continuously through a field sweep, for ANY Hamiltonian.

    build(B) must return a Hermitian matrix. Column k of the result is the same physical level at
    every field (ordered by energy at the first field). Tracking works by eigenvector overlap:
    after each step the new eigenvector most similar to an old one is the "same" level, so
    genuine crossings are followed straight through instead of being swapped by sorting.

    Returns energies (n_B, n) and, if return_vectors, also eigenvectors (n_B, n, n) whose columns
    follow the same ordering.
    """
    b_values = np.asarray(b_values, dtype=float)
    w, v = np.linalg.eigh(build(b_values[0]))
    n = w.size
    energies = np.empty((b_values.size, n))
    vectors = np.empty((b_values.size, n, n), dtype=complex) if return_vectors else None
    energies[0] = w
    if return_vectors:
        vectors[0] = v
    for i in range(1, b_values.size):
        w_new, v_new = np.linalg.eigh(build(b_values[i]))
        overlap = np.abs(v.conj().T @ v_new) ** 2           # (old, new)
        order = np.argmax(overlap, axis=1)
        if len(set(order)) < n:                              # ambiguous: fall back to sorted
            order = np.arange(n)
        energies[i] = w_new[order]
        v = v_new[:, order]
        if return_vectors:
            vectors[i] = v
    return (energies, vectors) if return_vectors else energies


def track_levels(l: int, b_values, zeta_ev: float, g_s: float = G_S) -> np.ndarray:
    """Tracked levels for one electron with orbital momentum l (see track_from_builder).

    Because H conserves J_z = m_l + m_s, levels with different J_z may cross but levels with the
    same J_z never do.
    """
    return track_from_builder(lambda b: h_total(l, b, zeta_ev, g_s=g_s), b_values)


def find_crossings(b_values, y_i, y_j) -> np.ndarray:
    """Field values where two tracked curves y_i(B) and y_j(B) cross (linear interpolation)."""
    b_values = np.asarray(b_values, dtype=float)
    d = np.asarray(y_i, dtype=float) - np.asarray(y_j, dtype=float)
    idx = np.where(d[:-1] * d[1:] < 0)[0]
    return b_values[idx] - d[idx] * (b_values[idx + 1] - b_values[idx]) / (d[idx + 1] - d[idx])


def zeeman_transitions(e_upper_ev, e_lower_ev):
    """All |upper - lower| energy differences (eV) between two level sets."""
    return np.abs(np.subtract.outer(np.asarray(e_upper_ev), np.asarray(e_lower_ev))).ravel()
