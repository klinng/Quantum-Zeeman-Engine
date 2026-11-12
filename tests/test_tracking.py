"""Eigenstate tracking: one-to-one assignment, crossings, avoided crossings, reversed sweeps."""
import warnings

import numpy as np
import pytest

from src.solvers import _assign, track_from_builder


def test_assign_is_one_to_one_where_argmax_is_not():
    # both old rows prefer new column 0 (greedy argmax would pick [0, 0])
    overlap = np.array([[0.55, 0.45], [0.60, 0.40]])
    assert list(np.argmax(overlap, axis=1)) == [0, 0]
    order = _assign(overlap)
    assert sorted(order) == [0, 1]
    assert list(order) == [1, 0]        # total 0.45+0.60 = 1.05 > 0.55+0.40 = 0.95


def test_genuine_crossing_is_followed_straight_through():
    b = np.linspace(-1, 1, 201)
    e = track_from_builder(lambda x: np.diag([x, -x, 5.0]), b)
    # level 0 started at -1 and keeps slope +1; level 1 keeps slope -1 (energy sorting would swap)
    assert np.allclose(e[:, 0], b) and np.allclose(e[:, 1], -b)
    assert np.allclose(e[:, 2], 5.0)


def test_avoided_crossing_follows_adiabatic_branches():
    delta = 0.3
    b = np.linspace(-2, 2, 801)
    e = track_from_builder(lambda x: np.array([[x, delta], [delta, -x]]), b)
    # coupled levels never cross: the two tracked branches are the sorted +-sqrt(x^2+delta^2)
    r = np.sqrt(b ** 2 + delta ** 2)
    assert np.allclose(np.sort(e, axis=1), np.c_[-r, r], atol=1e-12)
    assert np.all(e[:, 1] - e[:, 0] >= 2 * delta - 1e-12)


def test_reversed_sweep_gives_same_curves_for_a_crossing():
    b = np.linspace(-1, 1, 201)
    f = lambda x: np.diag([x, -x, 5.0])
    fwd = track_from_builder(f, b)
    rev = track_from_builder(f, b[::-1])[::-1]
    # forward starts with level 0 = lowest at B=-1 (slope +1); reversed starts at B=+1 where the
    # lowest level has slope -1, so the labels differ but the SET of curves is identical
    assert np.allclose(np.sort(fwd, axis=1), np.sort(rev, axis=1))
    assert np.allclose(rev[:, 0], -b) and np.allclose(rev[:, 1], b)


def test_near_degenerate_levels_energies_unaffected_by_basis_choice():
    # exactly degenerate pair at B=0 (arbitrary eigh basis); tiny splitting afterwards
    b = np.linspace(0, 1, 101)
    e = track_from_builder(lambda x: np.diag([1.0 + 1e-6 * x, 1.0 - 1e-6 * x, 2.0]), b)
    assert np.allclose(np.sort(e[:, :2], axis=1)[:, 1] - np.sort(e[:, :2], axis=1)[:, 0],
                       2e-6 * b, atol=1e-12)


def test_assignment_is_a_permutation_in_real_sweep():
    from src.hamiltonians import h_total
    b = np.linspace(0, 10, 401)
    _, v = track_from_builder(lambda x: h_total(2, x, 3e-4), b, return_vectors=True)
    for i in (1, 100, 400):                      # columns stay an orthonormal basis
        assert np.allclose(v[i].conj().T @ v[i], np.eye(10), atol=1e-10)


def test_coarse_grid_triggers_warning_and_confidence_is_reported():
    # a rotation by 0.7 rad per step: matched overlap cos^2 = 0.59 -> tracking is not trustworthy
    def build(x):
        c, s = np.cos(x), np.sin(x)
        r = np.array([[c, -s], [s, c]])
        return r @ np.diag([1.0, 2.0]) @ r.T
    with pytest.warns(RuntimeWarning, match="tracking uncertain"):
        track_from_builder(build, np.array([0.0, 0.7]), min_overlap=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, conf = track_from_builder(build, np.linspace(0, 0.5, 200), return_confidence=True)
    assert conf[0] == 1 and conf.min() > 0.99


# --- crossing detection and refinement (audit priority 5) ---------------------------------------

def test_crossing_exactly_on_a_grid_point_is_found_once():
    from src.solvers import find_crossings
    b = np.linspace(-1, 1, 201)                        # B = 0 is a grid point, d == 0 exactly there
    assert np.allclose(find_crossings(b, b, -b), [0.0])
    assert find_crossings(b, b, -b).size == 1          # strict d[k]*d[k+1] < 0 finds NONE of these


def test_tangential_touch_is_not_a_crossing_unless_requested():
    from src.solvers import find_crossings
    b = np.linspace(-1, 1, 201)
    y = b ** 2                                         # touches y = 0 at B = 0 without changing sign
    assert find_crossings(b, y, 0 * b).size == 0
    assert np.allclose(find_crossings(b, y, 0 * b, include_touching=True), [0.0])


def test_identical_curves_are_reported_once_not_everywhere():
    from src.solvers import find_crossings
    b = np.linspace(0, 1, 11)
    y = np.where(b < 0.5, b - 0.5, np.where(b > 0.7, b - 0.7, 0.0))   # zero over a plateau, sign change across it
    assert find_crossings(b, y, 0 * b).size == 1


def test_closely_spaced_crossings_both_found_when_grid_resolves_them():
    from src.solvers import find_crossings
    b = np.linspace(0, 1, 1001)
    y = (b - 0.500) * (b - 0.504)
    r = find_crossings(b, y, 0 * b)
    assert np.allclose(r, [0.500, 0.504], atol=2e-4)
    assert find_crossings(np.linspace(0, 1, 11), (np.linspace(0, 1, 11) - 0.500) * (np.linspace(0, 1, 11) - 0.504), 0).size == 0   # grid too coarse: silently blind


def test_refinement_beats_linear_interpolation_on_a_curved_difference():
    from src.solvers import find_crossings, refine_crossings_from_builder
    root = 0.37
    build = lambda x: np.diag([x * x * x + x, root ** 3 + root, 5.0])   # level 0 crosses constant level 1 at B=root
    b = np.linspace(0, 1, 6)
    lin = find_crossings(b, b ** 3 + b, np.full_like(b, root ** 3 + root))[0]
    ref = refine_crossings_from_builder(build, b, 0, 1, xtol=1e-13)[0]
    assert abs(ref - root) < 1e-12 < abs(lin - root)


def test_refined_crossing_in_a_real_hamiltonian_is_independent_of_the_grid():
    from src.hyperfine import lamb_crossings_gauss
    vals = [lamb_crossings_gauss(n=n, hyperfine=False)[0] for n in (6, 41)]
    assert max(vals) - min(vals) < 1e-7                # gauss; unrefined linear estimates differ by ~2e-4 G at n=21
    hf = [lamb_crossings_gauss(n=n) for n in (21, 81)]
    assert np.allclose(hf[0], hf[1], atol=1e-7) and len(hf[0]) == 4


def test_refined_crossing_is_a_true_degeneracy_of_the_exact_eigenvalues():
    """Independent of the tracking code: at the refined field two exact eigenvalues of H coincide (gap ~1e-17 eV,
    i.e. rounding), and the gap grows linearly (7.5e-5 eV/T) on either side, so a gap of g eV means the
    field is off by g/7.5e-5 T."""
    from src.hydrogen import h_n2
    from src.hyperfine import lamb_crossings_gauss
    bc = lamb_crossings_gauss(hyperfine=False)[0] * 1e-4
    w = np.linalg.eigvalsh(h_n2(bc))
    assert np.diff(w).min() < 1e-15                    # eV, numerically degenerate pair
    assert np.diff(np.linalg.eigvalsh(h_n2(bc + 1e-6))).min() == pytest.approx(7.54e-11, rel=0.01)
