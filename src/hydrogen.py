r"""Hydrogen n = 2: the 2s and 2p levels together, with fine structure and the Lamb shift.

The eight n = 2 states (spin included) are 2S1/2 (2 states), 2P1/2 (2 states) and 2P3/2 (4 states).
Both the spin-orbit term L.S and the Zeeman term conserve l, so the Hamiltonian is block diagonal:

    H = H(l=0) (+) H(l=1)

    H(l=0): 2S1/2 sits above 2P1/2 by the Lamb shift (a QED effect we take from experiment)
    H(l=1): spin-orbit zeta*L.S with zeta = (2P3/2 - 2P1/2 interval) / (3/2), plus the Zeeman term

Energies are measured from the 2P1/2 level at zero field. Consequence worth noticing: a magnetic
field alone never mixes 2s with 2p (l is conserved), so their levels CROSS rather than repel.
Mixing needs an electric field (the basis of the Lamb-Retherford experiment); not modelled here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import constants as sc
from scipy.linalg import block_diag

from .constants import G_S
from .hamiltonians import coupled_ops, h_total
from .solvers import track_from_builder

H_EV_PER_MHZ = sc.h / sc.e * 1e6        # energy of a 1 MHz photon, in eV
LAMB_SHIFT_MHZ = 1057.845               # E(2S1/2) - E(2P1/2) = 1057.845(9) MHz, Lundeen and Pipkin, quoted in arXiv:0809.2985
FINE_2P_MHZ = 10969.13                  # E(2P3/2) - E(2P1/2) = 10969.13(10) MHz, Baird et al., Phys. Rev. A 5, 564 (1972)


def mhz_to_ev(mhz: float) -> float:
    return mhz * H_EV_PER_MHZ


def ev_to_mhz(e_ev: float) -> float:
    return e_ev / H_EV_PER_MHZ


def dirac_fine_structure_mhz() -> float:
    """Pure Dirac-theory prediction of E(2P3/2) - E(2P1/2) = alpha^2 * Ry / 16, in MHz.

    A theory-only cross-check on FINE_2P_MHZ (they differ by ~0.2 % because of QED corrections).
    """
    ry_ev = sc.physical_constants["Rydberg constant times hc in eV"][0]
    return ev_to_mhz(sc.alpha ** 2 * ry_ev / 16)


def h_n2(b_tesla: float, fine_mhz: float = FINE_2P_MHZ, lamb_mhz: float = LAMB_SHIFT_MHZ,
         g_s: float = G_S) -> np.ndarray:
    """8x8 Hamiltonian of the n = 2 manifold. Basis: 2s (|ms=+1/2>,|ms=-1/2>) then 2p (6 states)."""
    zeta = mhz_to_ev(fine_mhz) / 1.5                    # 2p: E(3/2) - E(1/2) = (3/2) zeta
    h_s = h_total(0, b_tesla, 0.0, e0_ev=mhz_to_ev(lamb_mhz), g_s=g_s)
    h_p = h_total(1, b_tesla, zeta, e0_ev=zeta, g_s=g_s)   # e0 = zeta puts 2P1/2 at zero energy
    return block_diag(h_s, h_p)


def _operators():
    o0, o1 = coupled_ops(0), coupled_ops(1)
    j = {c: block_diag(o0["L" + c] + o0["S" + c], o1["L" + c] + o1["S" + c]) for c in "xyz"}
    j2 = sum(m @ m for m in j.values())
    return {"Jz": j["z"], "J2": j2, "P_s": block_diag(np.eye(2), np.zeros((6, 6)))}


def label_states(vectors: np.ndarray) -> pd.DataFrame:
    """Name each eigenvector (columns of `vectors`) by term and m_j.

    s_weight = probability of being in the 2s block; <J_z> gives m_j (exact, since J_z is conserved);
    <J^2> gives j (accurate while mu_B B is small compared with the fine structure).
    """
    ops = _operators()
    rows = []
    for k in range(vectors.shape[1]):
        v = vectors[:, k]
        s_w = float(np.real(v.conj() @ ops["P_s"] @ v))
        jz = float(np.real(v.conj() @ ops["Jz"] @ v))
        j2 = float(np.real(v.conj() @ ops["J2"] @ v))
        j = (-1 + np.sqrt(1 + 4 * j2)) / 2
        term = "2S1/2" if s_w > 0.5 else ("2P1/2" if j < 1.0 else "2P3/2")
        rows.append((k, term, round(2 * jz) / 2, s_w))
    return pd.DataFrame(rows, columns=["level", "term", "mj", "s_weight"])


def track_n2(b_values, **kw):
    """Tracked n = 2 levels over a field sweep.

    The first field must be small but positive (labels are read off there); it is raised to 1e-6 T
    if needed. Returns (b_values, energies in eV shape (n_B, 8), labels DataFrame, eigenvectors).
    """
    b = np.array(b_values, dtype=float)
    b[0] = max(b[0], 1e-6)
    energies, vectors = track_from_builder(lambda x: h_n2(x, **kw), b, return_vectors=True)
    return b, energies, label_states(vectors[0]), vectors


def level_index(labels: pd.DataFrame, term: str, mj: float) -> int:
    """Column of the tracked-energy array for a given term and m_j."""
    row = labels[(labels.term == term) & (labels.mj == mj)]
    if len(row) != 1:
        raise KeyError(f"no unique level for {term}, mj={mj}")
    return int(row.level.iloc[0])
