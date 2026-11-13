r"""Hyperfine structure: the nucleus (spin I) joins the electron (l, s).

Basis (uncoupled, used by every function in this module): |l m_l> (x) |s m_s> (x) |I m_I>, electron part first, nuclear
spin innermost; dimension 2(2l+1)(2I+1). The coupled basis |((l s) J, I) F m_F> is reached only through projectors
(electron_projectors for J, f_weights for F); F is a good quantum number at B = 0 only, m_F always (B along z).

    H = H_fine+Zeeman (x) 1_I  +  sum_J A_J  P_J (I.J) P_J  +  sum_J Q_J(B_J)  +  g_I mu_B B I_z

    Q_J(B_J) = B_J / (2I(2I-1)J(2J-1)) * P_J [ 3 (I.J)^2 + 3/2 (I.J) - I(I+1) J(J+1) ] P_J        (electric quadrupole)

The quadrupole term vanishes identically for J = 1/2 and for I = 1/2 (no quadrupole moment), so only sodium 3P3/2 needs it.

* P_J projects the electron space onto total electronic angular momentum J (built from the
  eigen-decomposition of J^2), so every fine-structure level J gets its OWN constant A_J.
* Nuclear Zeeman uses the convention H = mu_B g_I I_z B  (g_I in units of mu_B, negative for Na-23
  and for the proton), as in Steck's data sheets.
* The electric-quadrupole constant of sodium 3P3/2, B = 2.724(30) MHz (Yei, Sieradzan and Havey 1993), is included
  (NA_B_MHZ). Not modelled: the part of the magnetic hyperfine operator that couples different J (hydrogen 2P: at most 7 kHz
  at B = 0 and about 0.1 G on the 573 G crossings, estimated in the audit log), hyperfine-induced J mixing in sodium, and
  higher-order hyperfine constants (octupole).
* Ca-40 has I = 0: nothing to add. Ca-43 (I = 7/2) is not modelled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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
# Electric quadrupole constant of 3P3/2: B = 2.724(30) MHz, Yei, Sieradzan and Havey, Phys. Rev. A 48 (1993); the survey
# arXiv:2209.11423 lists the same value and a weighted average of 2.722(7) MHz. 3S1/2 and 3P1/2 have J = 1/2: no quadrupole term.
NA_B_MHZ = {"3P3/2": 2.724}
# Hydrogen, I = 1/2.
# 1S Fermi splitting 1420.405751768 MHz: arXiv:physics/0608192. 2S splitting 177 556 838.87(85) Hz: Phys. Rev. Lett. 130, 203001
# (Ramsey spectroscopy, abstract read; it supersedes the earlier 177.556860(16) MHz of arXiv:physics/0305073 by 21 Hz, 1.3 sigma of the old value). 2P1/2: theory value quoted in arXiv:1104.4421; 2P3/2: leading-order theory (Fermi/60), not checked against a source.
H_I = 0.5
H_FERMI_1S_MHZ = 1420.405751768
H_HFS_2S_MHZ = 177.55683887                 # measured, +-0.00000085 MHz
H_HFS_2S_SIGMA_MHZ = 0.85e-6
D21_THEORY_HZ, D21_THEORY_SIGMA_HZ = 48954.1, 2.3   # 8 f(2S) - f(1S), bound-state QED, quoted in the same abstract
H_HFS_2P12_MHZ = 59.2212                     # theory value quoted in arXiv:1104.4421 (measured there: 60(2) MHz); leading order alone gives Fermi/24 = 59.18
H_HFS_2P32_MHZ = H_FERMI_1S_MHZ / 60         # leading-order theory, about 23.67 MHz
H_G_I = -sc.physical_constants["proton g factor"][0] * sc.physical_constants["nuclear magneton"][0] \
    / sc.physical_constants["Bohr magneton"][0]


def hydrogen_hfs_leading_mhz(n: int, l: int, j: float) -> float:
    """Total hyperfine splitting (MHz) of hydrogen nlj at leading order (non-relativistic Fermi-Breit magnetic dipole), I = 1/2.

    A(nlj) = (3/8) nu_1S / (n^3 j (j+1) (l + 1/2)) in the convention H = A I.J, and the total splitting is A (I + j). It reproduces
    nS = nu_1S / n^3 and nP1/2 = nS / 3 exactly. Relativistic and radiative corrections, not included, change the S levels by
    about 1e-4 and the P1/2 level by about 1e-3 (see tests: 2S and 2P1/2 against the quoted values). The electric quadrupole
    constant B is exactly zero for hydrogen because the proton has spin 1/2. Not a measured or published table value."""
    a = 0.375 * H_FERMI_1S_MHZ / (n ** 3 * j * (j + 1) * (l + 0.5))
    return a * (H_I + j)


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


def quadrupole_operator(i: float, j: float, ij: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """Unit-B electric-quadrupole operator of one J level (in units of its constant B_J):
    [3 (I.J)^2 + 3/2 (I.J) - I(I+1)J(J+1)] / (2I(2I-1)J(2J-1)), restricted to the J subspace by `projector`.
    Zero for I < 1 or J < 1 (the denominator vanishes and the operator is identically zero there)."""
    if i < 1.0 or j < 1.0:
        return np.zeros_like(ij)
    pij = projector @ ij @ projector
    body = 3 * pij @ pij + 1.5 * pij - i * (i + 1) * j * (j + 1) * projector
    return body / (2 * i * (2 * i - 1) * j * (2 * j - 1))


def h_hyperfine(l: int, i: float, a_by_j_ev: dict, b_tesla: float, g_i: float, b_quad_by_j_ev: dict | None = None) -> np.ndarray:
    """Hyperfine (magnetic dipole, optional electric quadrupole) + nuclear Zeeman part, in the (l, s) (x) (I) space.
    b_quad_by_j_ev maps J to the quadrupole constant B_J in eV (omitted J = no quadrupole term)."""
    proj, jop = electron_projectors(l)
    ix, iy, iz = angular_momentum_ops(i)
    dim_e = 2 * (2 * l + 1)
    i_dot_j = sum(np.kron(jop[c], m) for c, m in zip("xyz", (ix, iy, iz)))
    h = np.zeros_like(i_dot_j, dtype=complex)
    for jv, p in proj.items():
        pj = np.kron(p, np.eye(int(round(2 * i + 1))))
        h += a_by_j_ev[jv] * (pj @ i_dot_j @ pj)
        if b_quad_by_j_ev and jv in b_quad_by_j_ev:
            h += b_quad_by_j_ev[jv] * quadrupole_operator(i, jv, i_dot_j, pj)
    h += MU_B_EV * g_i * b_tesla * np.kron(np.eye(dim_e), iz)
    return h


def h_with_hyperfine(l: int, i: float, a_by_j_ev: dict, b_tesla: float, g_i: float, zeta_ev: float,
                     e0_ev: float = 0.0, g_s: float = G_S, b_quad_by_j_ev: dict | None = None) -> np.ndarray:
    """Fine structure + electron Zeeman + hyperfine (+ quadrupole) + nuclear Zeeman for one l block."""
    h_e = h_total(l, b_tesla, zeta_ev, e0_ev=e0_ev, g_s=g_s)
    return np.kron(h_e, np.eye(int(round(2 * i + 1)))) + h_hyperfine(l, i, a_by_j_ev, b_tesla, g_i, b_quad_by_j_ev)


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


def h_na_3p(b_tesla: float, zeta_ev: float, quadrupole: bool = True) -> np.ndarray:
    """3P with I = 3/2 (24x24); zero of energy at 3P1/2 (e0 = zeta). quadrupole=True adds B(3P3/2) = NA_B_MHZ;
    quadrupole=False is the magnetic-dipole-only model (kept for comparison; it misplaces every 3P3/2 interval by 2.72 MHz)."""
    a = {0.5: mhz_to_ev(NA_A_MHZ["3P1/2"]), 1.5: mhz_to_ev(NA_A_MHZ["3P3/2"])}
    q = {1.5: mhz_to_ev(NA_B_MHZ["3P3/2"])} if quadrupole else None
    return h_with_hyperfine(1, NA_I, a, b_tesla, NA_G_I, zeta_ev, e0_ev=zeta_ev, b_quad_by_j_ev=q)


def f_weights(h_hf: np.ndarray, l: int, i: float, eigenvectors: np.ndarray) -> pd.DataFrame:
    """Character of hyperfine eigenstates: for each column of `eigenvectors` (eigenstates of h_hf, sorted as returned by eigh)
    the weights in the J subspaces (p_j*) and in the F subspaces of the (l, s) (x) (I) space, the exact m_F = <J_z + I_z>
    rounded (B along z), the dominant F and its purity. F is a good label at B = 0 only; in a field use m_F plus the weights
    (or follow the level from B -> 0). The F subspaces are the spectral projectors of F^2 = (J + I)^2."""
    import pandas as pd
    proj, jop = electron_projectors(l)
    ix, iy, iz = angular_momentum_ops(i)
    dim_i = int(round(2 * i + 1))
    f = {c: np.kron(jop[c], np.eye(dim_i)) + np.kron(np.eye(jop["z"].shape[0]), m) for c, m in zip("xyz", (ix, iy, iz))}
    f2 = sum(m @ m for m in f.values())
    w, u = np.linalg.eigh(f2)
    f_vals = sorted({round((-1 + np.sqrt(1 + 4 * x)) / 2 * 2) / 2 for x in w.real})
    pf = {fv: u[:, np.isclose(w.real, fv * (fv + 1), atol=1e-6)] @ u[:, np.isclose(w.real, fv * (fv + 1), atol=1e-6)].conj().T
          for fv in f_vals}
    pj = {jv: np.kron(p, np.eye(dim_i)) for jv, p in proj.items()}
    rows = []
    for k in range(eigenvectors.shape[1]):
        v = eigenvectors[:, k]
        ex = lambda m: float(np.real(v.conj() @ m @ v))
        wf = {fv: ex(p) for fv, p in pf.items()}
        fdom = max(wf, key=wf.get)
        row = {"m_f": round(ex(f["z"]), 6), "f": fdom, "f_purity": wf[fdom]}
        row.update({f"p_j{jv:g}": ex(p) for jv, p in pj.items()})
        row.update({f"p_f{fv:g}": wf[fv] for fv in f_vals})
        rows.append(row)
    return pd.DataFrame(rows)


# ---- analysis helpers -----------------------------------------------------------------------
def lamb_crossings_gauss(b_window_t=(0.050, 0.065), n: int = 301, hyperfine: bool = True,
                         refine: bool = True) -> list[float]:
    """Fields (gauss) where a 2S level crosses a 2P level in the window around 573 G.

    Without hyperfine there is a single crossing; with hyperfine the proton spin splits it into four.
    refine=True (default) re-diagonalises H inside each bracket until it is narrower than 1e-12 T (see
    solvers.refine_crossings_from_builder), so the result does not depend on n; refine=False gives the
    linear-interpolation estimate from the grid.
    """
    from .hydrogen import h_n2
    from .solvers import find_crossings, refine_crossings_from_builder, track_from_builder
    bs = np.linspace(*b_window_t, n)
    build = h_n2_hyperfine if hyperfine else h_n2
    n_s = 4 if hyperfine else 2
    e, v = track_from_builder(build, bs, return_vectors=True)
    s_w = np.sum(np.abs(v[0][:n_s, :]) ** 2, axis=0)
    s_idx, p_idx = np.where(s_w > 0.5)[0], np.where(s_w <= 0.5)[0]
    out = []
    for a in s_idx:
        for c in p_idx:
            if refine and np.any(np.diff(np.sign(e[:, a] - e[:, c])) != 0):
                roots = refine_crossings_from_builder(build, bs, int(a), int(c), xtol=1e-12, tracked=(e, v))
            else:
                roots = find_crossings(bs, e[:, a], e[:, c])
            out += [float(x) * 1e4 for x in np.atleast_1d(roots)]
    return sorted(out)
