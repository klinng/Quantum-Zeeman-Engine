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


def find_crossings(b_values, y_i, y_j, include_touching: bool = False) -> np.ndarray:
    """Field values where two tracked curves y_i(B) and y_j(B) cross (linear interpolation in each interval).

    Handles three cases the plain "sign change between neighbours" test gets wrong:
      * a crossing that lands EXACTLY on a grid point (d == 0): reported once, at that point, if d changes
        sign across it (the strict test d[k]*d[k+1] < 0 misses it and nobody notices);
      * a run of exact zeros (identical curves): reported once at its first point;
      * a tangential touch (d reaches zero without changing sign): not a crossing; returned only with
        include_touching=True.
    Linear interpolation is only a first estimate (error ~ curvature * h^2); for a final value use
    refine_crossings_from_builder, which re-diagonalises inside the bracket.
    """
    b = np.asarray(b_values, dtype=float)
    d = np.asarray(y_i, dtype=float) - np.asarray(y_j, dtype=float)
    sgn = np.sign(d)
    out = []
    k = 0
    while k < d.size:
        if sgn[k] == 0:
            m = k                                           # extent of the zero run [k, m]
            while m + 1 < d.size and sgn[m + 1] == 0:
                m += 1
            before = sgn[k - 1] if k > 0 else 0
            after = sgn[m + 1] if m + 1 < d.size else 0
            if before * after < 0 or (include_touching and (before == 0 or after == 0 or before == after)):
                out.append(b[k])
            k = m + 1
        else:
            if k + 1 < d.size and sgn[k + 1] * sgn[k] < 0:
                out.append(b[k] - d[k] * (b[k + 1] - b[k]) / (d[k + 1] - d[k]))
            k += 1
    return np.array(out, dtype=float)


def refine_crossings_from_builder(build, b_values, i: int, j: int, xtol: float = 1e-12, n_sub: int = 9,
                                  max_iter: int = 60, tracked=None) -> np.ndarray:
    """Crossings of tracked levels i and j with the Hamiltonian RE-DIAGONALISED inside each bracket.

    A first pass on b_values brackets each sign change of E_i - E_j. Each bracket is then zoomed: the
    Hamiltonian is re-tracked on n_sub points inside it (track_from_builder, so level identity is kept by
    overlap, not by energy order) and the sub-interval containing the sign change becomes the new bracket,
    until its width is below xtol (in the units of b_values). tracked=(energies, vectors) from a previous
    track_from_builder(..., return_vectors=True) call on the same grid avoids repeating the first sweep. Nothing is interpolated at the end: the
    returned field is the midpoint of a bracket whose endpoints were both diagonalised exactly.
    """
    b = np.asarray(b_values, dtype=float)
    e0, v0 = tracked if tracked is not None else track_from_builder(build, b, return_vectors=True)
    d0 = e0[:, i] - e0[:, j]
    roots = []
    for k in np.where(d0[:-1] * d0[1:] < 0)[0]:
        lo, hi = b[k], b[k + 1]
        v_lo = v0[k]
        for _ in range(max_iter):
            if hi - lo <= xtol:
                break
            sub = np.linspace(lo, hi, n_sub)
            # columns of the sub-sweep follow the energy order at sub[0]; order[k] = the column that is the
            # same physical level as global level k (matched by overlap with the eigenvectors at lo)
            _, v_first = np.linalg.eigh(build(sub[0]))
            order = _assign(np.abs(v_lo.conj().T @ v_first) ** 2)
            en, vs = track_from_builder(build, sub, return_vectors=True, min_overlap=0.0)
            dd = en[:, order[i]] - en[:, order[j]]
            hit = np.where(dd[:-1] * dd[1:] <= 0)[0]
            if hit.size == 0:
                break
            m = hit[0]
            lo, hi = sub[m], sub[m + 1]
            v_lo = vs[m][:, order]
        roots.append(0.5 * (lo + hi))
    return np.array(roots)


def zeeman_transitions(e_upper_ev, e_lower_ev):
    """All |upper - lower| energy differences (eV) between two level sets."""
    return np.abs(np.subtract.outer(np.asarray(e_upper_ev), np.asarray(e_lower_ev))).ravel()
