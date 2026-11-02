"""Hamiltonian Hermiticity, angular-momentum algebra and unitarity of the eigenbasis."""
import numpy as np
import pytest

from src.hamiltonians import angular_momentum_ops, coupled_ops, h_total
from src.solvers import solve


@pytest.mark.parametrize("j", [0.5, 1, 1.5, 2])
def test_commutation_relations(j):
    jx, jy, jz = angular_momentum_ops(j)
    assert jx @ jy - jy @ jx == pytest.approx(1j * jz, abs=1e-12)
    j2 = jx @ jx + jy @ jy + jz @ jz
    assert j2 == pytest.approx(j * (j + 1) * np.eye(jz.shape[0]), abs=1e-12)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
@pytest.mark.parametrize("b", [0.0, 0.5, 10.0])
def test_hermitian(l, b):
    h = h_total(l, b, 1e-3)
    assert np.allclose(h, h.conj().T, atol=1e-15)


def test_eigenvectors_unitary():
    _, v = solve(2, 3.0, 1e-3)
    assert v.conj().T @ v == pytest.approx(np.eye(v.shape[0]), abs=1e-12)


def test_jz_conserved():
    """[H, J_z] = 0 where J_z = L_z + S_z."""
    o = coupled_ops(2)
    jz = o["Lz"] + o["Sz"]
    h = h_total(2, 4.0, 1e-3)
    assert h @ jz - jz @ h == pytest.approx(0, abs=1e-15)
