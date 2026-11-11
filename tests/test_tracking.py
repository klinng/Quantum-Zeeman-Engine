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
