import numpy as np
import pytest

from src.cavity import (annihilation, creation, excitation_number, excited_probability, jaynes_cummings)


def test_commutator_is_one_except_on_the_last_state():
    n = 8
    comm = annihilation(n) @ creation(n) - creation(n) @ annihilation(n)
    assert np.allclose(np.diag(comm)[:-1], 1.0) and np.allclose(comm - np.diag(np.diag(comm)), 0)


def test_number_operator_counts_photons():
    n = 6
    assert np.allclose(np.diag(creation(n) @ annihilation(n)), np.arange(n + 1))


def test_excitation_number_is_conserved():
    h = jaynes_cummings(10, 10.0, 9.0, 0.7)
    num = excitation_number(10)
    assert np.allclose(h @ num, num @ h, atol=1e-12)


@pytest.mark.parametrize("big_n", [1, 2, 3])
def test_resonant_levels_split_by_g_sqrt_n(big_n):
    g, wc = 0.8, 10.0
    h = jaynes_cummings(10, wc, wc, g)
    idx = [2 * (big_n - 1), 2 * big_n + 1]                       # |N-1, excited> and |N, ground>
    e = np.linalg.eigvalsh(h[np.ix_(idx, idx)])
    assert e == pytest.approx([(big_n - 0.5) * wc - g * np.sqrt(big_n), (big_n - 0.5) * wc + g * np.sqrt(big_n)], abs=1e-9)


def test_vacuum_rabi_oscillation_is_cos_squared():
    t = np.linspace(0, 10, 50)
    assert np.allclose(excited_probability(8, 10.0, 10.0, 1.3, t), np.cos(1.3 * t) ** 2, atol=1e-9)


def test_detuned_rabi_formula():
    g, delta = 1.0, 3.0
    t = np.linspace(0, 8, 40)
    omega = np.sqrt(delta ** 2 + 4 * g ** 2)
    expected = 1 - 4 * g ** 2 / omega ** 2 * np.sin(omega * t / 2) ** 2
    assert np.allclose(excited_probability(8, 10.0, 10.0 + delta, g, t), expected, atol=1e-9)
