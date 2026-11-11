"""Numerical eigensolvers for the Zeeman Hamiltonians."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def _assign(overlap: np.ndarray) -> np.ndarray:
    """One-to-one old->new assignment maximising the total overlap (Hungarian algorithm).

    A per-row argmax can send two old levels to the same new eigenvector; the global assignment
    cannot. Returns order such that new level order[k] continues old level k.
    """
    rows, cols = linear_sum_assignment(-overlap)
    order = np.empty(overlap.shape[0], dtype=int)
    order[rows] = cols
    return order


def track_from_builder(build, b_values, return_vectors: bool = False, min_overlap: float = 0.5,
                       return_confidence: bool = False):
    """Follow every energy level continuously through a field sweep, for ANY Hamiltonian.

    build(B) must return a Hermitian matrix. Column k of the result is the same physical level at
    every field (ordered by energy at the first field). Tracking works by eigenvector overlap:
    after each step the new eigenvector most similar to an old one is the "same" level, so
    genuine crossings are followed straight through instead of being swapped by sorting.

    The old->new matching is a global one-to-one assignment (scipy linear_sum_assignment on the
    overlap matrix), so no new eigenvector is ever claimed twice. Inside an exactly degenerate
    subspace the eigenvectors returned by eigh are an arbitrary basis; the assignment is then
    only defined up to a rotation of that subspace, but the ENERGIES are unaffected. If the best
    matched overlap at any step falls below min_overlap the step is too coarse (or the levels are
    nearly degenerate and strongly mixing) and a RuntimeWarning is issued.

    Returns energies (n_B, n) and, if return_vectors, also eigenvectors (n_B, n, n) whose columns
    follow the same ordering. With return_confidence=True a final array (n_B,) of the smallest
    matched overlap per step is appended (1 at the first field).
    """
    b_values = np.asarray(b_values, dtype=float)
    w, v = np.linalg.eigh(build(b_values[0]))
    n = w.size
    energies = np.empty((b_values.size, n))
    vectors = np.empty((b_values.size, n, n), dtype=complex) if return_vectors else None
    energies[0] = w
    confidence = np.ones(b_values.size)
    if return_vectors:
        vectors[0] = v
    for i in range(1, b_values.size):
        w_new, v_new = np.linalg.eigh(build(b_values[i]))
        overlap = np.abs(v.conj().T @ v_new) ** 2           # (old, new)
        order = _assign(overlap)
        confidence[i] = overlap[np.arange(n), order].min()
        if confidence[i] < min_overlap:
            warnings.warn(f"eigenstate tracking uncertain at B={b_values[i]:.6g} "
                          f"(smallest matched overlap {confidence[i]:.3f}); refine the B grid",
                          RuntimeWarning, stacklevel=2)
        energies[i] = w_new[order]
        v = v_new[:, order]
        if return_vectors:
            vectors[i] = v
    out = (energies, vectors) if return_vectors else (energies,)
    if return_confidence:
        out = out + (confidence,)
    return out if len(out) > 1 else out[0]


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
