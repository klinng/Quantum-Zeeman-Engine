r"""Hydrogen n = 2 in crossed magnetic and electric fields: Stark mixing and the Lamb-Retherford anticrossing.

A magnetic field never mixes 2s with 2p (src/hydrogen.py), so the levels 2S1/2(m_j=-1/2) and 2P1/2(m_j=+1/2) CROSS near 573 G.
An electric field changes that. The Stark term  H_E = e E . r  is odd under parity, so it connects 2s with 2p. Its only
n = 2 matrix elements are (a0 = Bohr radius, signs are convention)

    <2s| z |2p, m=0>  = -3 a0            (field along B: conserves m_j)
    <2s| x |2p, m=+-1> = +-3 a0 / sqrt(2) (field perpendicular to B: changes m_j by 1)

With E perpendicular to B the two crossing levels differ by one unit of m_j, so the field couples them and the
crossing becomes an anticrossing of width (gap) 2 sqrt(3) e a0 E = 4.43 MHz per V/cm  (1 e a0 = 1.28 MHz per V/cm).
That mixing is what quenches the metastable 2S state in the Lamb-Retherford experiment.
Not modelled: hyperfine structure, the diamagnetic term, and the 2p lifetime width (the gap is a level repulsion,
not a decay rate).
"""
from __future__ import annotations

import numpy as np
from scipy import constants as sc
from scipy.linalg import block_diag

from .hamiltonians import coupled_ops
from .hydrogen import h_n2, mhz_to_ev
from .solvers import track_from_builder

E_A0_MHZ_PER_VCM = sc.e * sc.physical_constants["Bohr radius"][0] * 100.0 / sc.h / 1e6   # e a0 x (1 V/cm), in MHz
GAP_MHZ_PER_VCM = 2 * np.sqrt(3) * E_A0_MHZ_PER_VCM                                       # anticrossing width per V/cm


def _p_index(ml: int, ms: float) -> int:
    o = coupled_ops(1)
    lz, sz = np.diag(o["Lz"]).real, np.diag(o["Sz"]).real
    hit = np.where((np.abs(lz - ml) < 1e-9) & (np.abs(sz - ms) < 1e-9))[0]
    return int(hit[0])


def _s_index(ms: float) -> int:
    sz = np.diag(coupled_ops(0)["Sz"]).real
    return int(np.where(np.abs(sz - ms) < 1e-9)[0][0])


def stark_matrix(direction: str = "perp") -> np.ndarray:
    """8x8 matrix of e E . r in units of (e a0 E). Basis as in hydrogen.h_n2: 2s (2 states) then 2p (6 states).

    direction 'par': E along B (z); 'perp': E perpendicular to B (x).
    """
    m = np.zeros((8, 8), dtype=complex)
    for ms in (0.5, -0.5):
        s = _s_index(ms)
        if direction == "par":
            m[s, 2 + _p_index(0, ms)] = -3.0
        elif direction == "perp":
            m[s, 2 + _p_index(+1, ms)] = +3.0 / np.sqrt(2)
            m[s, 2 + _p_index(-1, ms)] = -3.0 / np.sqrt(2)
        else:
            raise ValueError("direction must be 'par' or 'perp'")
    return m + m.conj().T


def h_n2_stark(b_tesla: float, e_vcm: float, direction: str = "perp", **kw) -> np.ndarray:
    """n = 2 Hamiltonian (eV) in a magnetic field B (T) and an electric field E (V/cm)."""
    return h_n2(b_tesla, **kw) + mhz_to_ev(E_A0_MHZ_PER_VCM * e_vcm) * stark_matrix(direction)


def track_stark(b_values, e_vcm: float, direction: str = "perp"):
    """Tracked levels (eV, shape (n_B, 8)) and eigenvectors over a field sweep."""
    b = np.array(b_values, dtype=float)
    b[0] = max(b[0], 1e-6)
    en, vec = track_from_builder(lambda x: h_n2_stark(x, e_vcm, direction), b, return_vectors=True)
    return b, en, vec


def crossing_gap_mhz(e_vcm: float, direction: str = "perp", b_lo: float = 0.04, b_hi: float = 0.075,
                     n_points: int = 600) -> float:
    """Smallest distance (MHz) between the two levels that cross near 573 G, scanning B (T).

    The pair is the two levels whose energy difference is smallest inside the window (excluding other crossings by
    taking the 2S1/2(m_j = -1/2) and 2P1/2(m_j = +1/2) states labelled at weak field).
    """
    from .hydrogen import label_states, level_index
    b = np.linspace(b_lo, b_hi, n_points)
    en, vec = track_from_builder(lambda x: h_n2_stark(x, e_vcm, direction), b, return_vectors=True)
    # labels are read off from the field-free eigenvectors of the pure magnetic problem
    from .solvers import track_from_builder as _t
    b0, e0, lab, v0 = (lambda bb: (bb, *_t(lambda x: h_n2(x), bb, return_vectors=True)))(np.array([1e-6, b_lo, b_hi]))
    lab = label_states(v0[0])
    s_idx, p_idx = level_index(lab, "2S1/2", -0.5), level_index(lab, "2P1/2", 0.5)
    # follow the two states by their energies relative to the pure-B crossing: use the pair with minimal separation
    # among the level pairs that are closest at the window centre
    mid = len(b) // 2
    pair = sorted(range(8), key=lambda k: abs(en[mid, k] - en[mid, s_idx]))[:2]
    return float(np.min(np.abs(en[:, pair[0]] - en[:, pair[1]])) / mhz_to_ev(1.0))
