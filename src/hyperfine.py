r"""Hyperfine structure: the nucleus (spin I) joins the electron (l, s).

Basis |l m_l> (x) |s m_s> (x) |I m_I>  (electron part first, nuclear spin innermost).

    H = H_fine+Zeeman (x) 1_I  +  sum_J A_J  P_J (I.J) P_J  +  g_I mu_B B I_z

* P_J projects the electron space onto total electronic angular momentum J (built from the
  eigen-decomposition of J^2), so every fine-structure level J gets its OWN constant A_J.
* Nuclear Zeeman uses the convention H = mu_B g_I I_z B  (g_I in units of mu_B, negative for Na-23
  and for the proton), as in Steck's data sheets.
* The electric-quadrupole constant B(3P3/2) of sodium (2.7 MHz) is NOT modelled.
* Ca-40 has I = 0: nothing to add.
"""
from __future__ import annotations

import numpy as np
from scipy import constants as sc
from scipy.linalg import block_diag

from .constants import MU_B_EV, G_S
from .hamiltonians import angular_momentum_ops, coupled_ops, h_total
from .hydrogen import LAMB_SHIFT_MHZ, FINE_2P_MHZ, mhz_to_ev

# ---- constants (MHz) ----------------------------------------------------------------------------
# Sodium-23, I = 3/2. Source: D. A. Steck, "Sodium D Line Data", hyperfine table
# (A(3S1/2) = 885.8130644(5), A(3P1/2) = 94.44(13), A(3P3/2) = 18.534(15) MHz; g_I = -0.0008046108(8)).
NA_I = 1.5
NA_A_MHZ = {"3S1/2": 885.8130644, "3P1/2": 94.44, "3P3/2": 18.534}
NA_G_I = -0.0008046108
# Hydrogen, I = 1/2.
# 1S Fermi splitting 1420.405751768 MHz and 2S splitting 177.556860(16) MHz: arXiv:physics/0608192;
# (arXiv:physics/0608192 and arXiv:physics/0305073, measured). 2P1/2: theory value quoted in arXiv:1104.4421; 2P3/2: leading-order theory (Fermi/60), not checked against a source.
H_I = 0.5
H_FERMI_1S_MHZ = 1420.405751768
H_HFS_2S_MHZ = 177.556860                    # measured
H_HFS_2P12_MHZ = 59.2212                     # theory value quoted in arXiv:1104.4421 (measured there: 60(2) MHz); leading order alone gives Fermi/24 = 59.18
H_HFS_2P32_MHZ = H_FERMI_1S_MHZ / 60         # leading-order theory, about 23.67 MHz
H_G_I = -sc.physical_constants["proton g factor"][0] * sc.physical_constants["nuclear magneton"][0] \
    / sc.physical_constants["Bohr magneton"][0]


def a_from_splitting_mhz(total_splitting_mhz: float, j: float, i: float) -> float:
    """A from the zero-field splitting between F = J+I and F = J+I-1: Delta = A (J + I) ... = A*(F_max)."""
    return total_splitting_mhz / (j + i)


def electron_projectors(l: int):
    """Projectors onto each J = |l-1/2|, l+1/2 in the (l, s) space. Returns {j: P}."""
    o = coupled_ops(l)
    j = {c: o["L" + c] + o["S" + c] for c in "xyz"}
    j2 = sum(m @ m for m in j.values())
    w, v = np.linalg.eigh(j2)
    out = {}
    for jv in sorted({round((-1 + np.sqrt(1 + 4 * x)) / 2 * 2) / 2 for x in w.real}):
        sel = np.isclose(w.real, jv * (jv + 1), atol=1e-6)
        out[jv] = v[:, sel] @ v[:, sel].conj().T
    return out, j


def h_hyperfine(l: int, i: float, a_by_j_ev: dict, b_tesla: float, g_i: float) -> np.ndarray:
    """Hyperfine + nuclear Zeeman part only, in the (l, s) (x) (I) space."""
    proj, jop = electron_projectors(l)
    ix, iy, iz = angular_momentum_ops(i)
    dim_e = 2 * (2 * l + 1)
    i_dot_j = sum(np.kron(jop[c], m) for c, m in zip("xyz", (ix, iy, iz)))
    h = np.zeros_like(i_dot_j, dtype=complex)
    for jv, p in proj.items():
        pj = np.kron(p, np.eye(int(round(2 * i + 1))))
        h += a_by_j_ev[jv] * (pj @ i_dot_j @ pj)
    h += MU_B_EV * g_i * b_tesla * np.kron(np.eye(dim_e), iz)
    return h


def h_with_hyperfine(l: int, i: float, a_by_j_ev: dict, b_tesla: float, g_i: float, zeta_ev: float,
                     e0_ev: float = 0.0, g_s: float = G_S) -> np.ndarray:
    """Fine structure + electron Zeeman + hyperfine + nuclear Zeeman for one l block."""
    h_e = h_total(l, b_tesla, zeta_ev, e0_ev=e0_ev, g_s=g_s)
    return np.kron(h_e, np.eye(int(round(2 * i + 1)))) + h_hyperfine(l, i, a_by_j_ev, b_tesla, g_i)


# ---- hydrogen n = 2 ---------------------------------------------------------------------------
def h_n2_hyperfine(b_tesla: float, hfs_2s=H_HFS_2S_MHZ, hfs_2p12=H_HFS_2P12_MHZ, hfs_2p32=H_HFS_2P32_MHZ,
                   fine_mhz: float = FINE_2P_MHZ, lamb_mhz: float = LAMB_SHIFT_MHZ) -> np.ndarray:
    """16x16 Hamiltonian of hydrogen n = 2 with the proton spin. Basis: 2s (4 states) then 2p (12)."""
    zeta = mhz_to_ev(fine_mhz) / 1.5
    a_s = {0.5: mhz_to_ev(a_from_splitting_mhz(hfs_2s, 0.5, H_I))}
    a_p = {0.5: mhz_to_ev(a_from_splitting_mhz(hfs_2p12, 0.5, H_I)),
           1.5: mhz_to_ev(a_from_splitting_mhz(hfs_2p32, 1.5, H_I))}
    h_s = h_with_hyperfine(0, H_I, a_s, b_tesla, H_G_I, 0.0, e0_ev=mhz_to_ev(lamb_mhz))
    h_p = h_with_hyperfine(1, H_I, a_p, b_tesla, H_G_I, zeta, e0_ev=zeta)
    return block_diag(h_s, h_p)


# ---- sodium ----------------------------------------------------------------------------------
def h_na_3s(b_tesla: float) -> np.ndarray:
    """3S1/2 with I = 3/2 (8x8)."""
    a = {0.5: mhz_to_ev(NA_A_MHZ["3S1/2"])}
    return h_with_hyperfine(0, NA_I, a, b_tesla, NA_G_I, 0.0)


def h_na_3p(b_tesla: float, zeta_ev: float) -> np.ndarray:
    """3P with I = 3/2 (24x24); zero of energy at 3P1/2 (e0 = zeta)."""
    a = {0.5: mhz_to_ev(NA_A_MHZ["3P1/2"]), 1.5: mhz_to_ev(NA_A_MHZ["3P3/2"])}
    return h_with_hyperfine(1, NA_I, a, b_tesla, NA_G_I, zeta_ev, e0_ev=zeta_ev)


# ---- analysis helpers -----------------------------------------------------------------------
def lamb_crossings_gauss(b_window_t=(0.050, 0.065), n: int = 301, hyperfine: bool = True) -> list[float]:
    """Fields (gauss) where a 2S level crosses a 2P level in the window around 573 G.

    Without hyperfine there is a single crossing; with hyperfine the proton spin splits it into four.
    """
    from .hydrogen import h_n2
    from .solvers import find_crossings, track_from_builder
    bs = np.linspace(*b_window_t, n)
    build = h_n2_hyperfine if hyperfine else h_n2
    n_s = 4 if hyperfine else 2
    e, v = track_from_builder(build, bs, return_vectors=True)
    s_w = np.sum(np.abs(v[0][:n_s, :]) ** 2, axis=0)
    s_idx, p_idx = np.where(s_w > 0.5)[0], np.where(s_w <= 0.5)[0]
    out = []
    for a in s_idx:
        for c in p_idx:
            out += [float(x) * 1e4 for x in np.atleast_1d(find_crossings(bs, e[:, a], e[:, c]))]
    return sorted(out)
