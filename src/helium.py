r"""Helium 1s2s and 1s2p: exchange from Slater integrals, spin-orbit, spin-spin, and the magnetic field.

Electron 1 stays in 1s (l = 0), electron 2 is in 2s or 2p. In the uncoupled basis |m_l> x |m_s1> x |m_s2> of electron 2's
orbital and the two spins, the Hamiltonian is built in this order (B is added last, so everything before it is the zero-field
level structure):

    H = E_dir                                  direct (Coulomb) energy F^0 + the configuration energy (a constant)
      - G (1/2 + 2 s1.s2)                      EXCHANGE: triplet -G, singlet +G, so E(singlet) - E(triplet) = 2 G^k
      + zeta_S  L.S + zeta_A L.(s2 - s1)       spin-orbit and spin-other-orbit (zeta_A mixes singlet and triplet)
      + rho [ (L.S)^2 + (1/2) L.S - (1/3) L^2 S^2 ]   spin-spin (tensor) interaction, S = s1 + s2, L = l2
      + mu_B B (L_z + g_s S_z)                 Zeeman.

* k = l: the exchange integral is G^0(1s,2s) for 1s2s (4 states, spin only) and G^1(1s,2p) for 1s2p (12 states).
* exchange: for two electrons the exchange operator is -G (1/2 + 2 s1.s2), which equals -G for S = 1 (s1.s2 = 1/4) and +G for
  S = 0 (s1.s2 = -3/4). This is two_electron.py's Delta (1/4 - s1.s2) with Delta = 2 G, up to a constant.
* The spin-spin operator is a rank-2 tensor in spin and in orbit: it is zero in singlets (S = 0) and in any S level
  averaged over J, and it gives the 2^3P_J pattern 1 : -5 : 10 for J = 2, 1, 0 (times rho/6). It is what breaks the Lande
  interval rule in helium; the spin-orbit term alone gives the 2 : 1 rule. Spin-spin and spin-other-orbit are taken as
  effective parameters here (fitted to the measured 2^3P intervals), not computed from the two-electron integrals.
* Slater integrals: ``slater_integral`` evaluates G^k and F^k numerically from hydrogenic orbitals with effective charges.
  Those orbitals are a crude model of helium (the result is an estimate of the order of magnitude, not a precise G); the
  precise G^k is best taken from the measured singlet-triplet splitting: G = (E_singlet - E_triplet) / 2.
Not modelled: configuration mixing (1s2p with 1s3p, 2s2p), nuclear spin (He-3 hyperfine), QED corrections. Energies in eV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from .constants import G_S, MU_B_EV
from .two_electron import operators

HARTREE_EV = 27.211386245988
_I12 = np.eye(12)


def _ops():
    o = operators()
    s2sq = sum(o["S" + c] @ o["S" + c] for c in "xyz")
    ls = sum(o["L" + c] @ o["S" + c] for c in "xyz")
    l_dot_s2 = sum(o["L" + c] @ o["s2" + c] for c in "xyz")
    l_dot_s1 = sum(o["L" + c] @ o["s1" + c] for c in "xyz")
    s1s2 = sum(o["s1" + c] @ o["s2" + c] for c in "xyz")
    return o, s2sq, ls, l_dot_s2 - l_dot_s1, s1s2


def h_exchange(g_ev: float) -> np.ndarray:
    """Exchange operator -G (1/2 + 2 s1.s2) on the 12-state space: -G in the triplet, +G in the singlet."""
    return -g_ev * (0.5 * _I12 + 2 * _ops()[4])


def h_spin_spin(rho_ev: float) -> np.ndarray:
    """Effective spin-spin tensor rho [(L.S)^2 + L.S/2 - L^2 S^2/3]; zero for S = 0, 1 : -5 : 10 over J = 2, 1, 0 in the triplet (x rho/6)."""
    o, s2sq, ls, _, _ = _ops()
    l2 = sum(o["L" + c] @ o["L" + c] for c in "xyz")
    return rho_ev * (ls @ ls + 0.5 * ls - l2 @ s2sq / 3.0)


def h_helium_1s2p(b_tesla: float, g_ev: float, zeta_s_ev: float, rho_ev: float, zeta_a_ev: float = 0.0,
                  e_dir_ev: float = 0.0, g_s: float = G_S) -> np.ndarray:
    """12 x 12 Hamiltonian of 1s2p: direct + exchange(G^1) + spin-orbit + spin-spin, then Zeeman. Energies in eV."""
    o, s2sq, ls, l_dot_diff, _ = _ops()
    h0 = e_dir_ev * _I12 + h_exchange(g_ev) + zeta_s_ev * ls + zeta_a_ev * l_dot_diff + h_spin_spin(rho_ev)
    return h0 + MU_B_EV * b_tesla * (o["Lz"] + g_s * o["Sz"])


def h_helium_1s2s(b_tesla: float, g_ev: float, e_dir_ev: float = 0.0, g_s: float = G_S) -> np.ndarray:
    """4 x 4 Hamiltonian of 1s2s on |m_s1> x |m_s2>: direct + exchange(G^0) + Zeeman (S_z only, since L = 0)."""
    sx = np.array([[0, 0.5], [0.5, 0]]); sy = np.array([[0, -0.5j], [0.5j, 0]]); sz = np.diag([0.5, -0.5])
    i2 = np.eye(2)
    s1s2 = sum(np.kron(s, s) for s in (sx, sy, sz))
    stot_z = np.kron(sz, i2) + np.kron(i2, sz)
    return e_dir_ev * np.eye(4) - g_ev * (0.5 * np.eye(4) + 2 * s1s2) + MU_B_EV * b_tesla * g_s * stot_z


def triplet_p_levels(zeta_s_ev: float, rho_ev: float, e_ev: float = 0.0) -> dict[int, float]:
    """The three 2^3P_J energies at zero field from the closed form E_J = e + zeta_S (L.S)_J + rho f_J, (L.S)_J = 1, -1, -2 and f_J = 1/6, -5/6, 5/3 for J = 2, 1, 0."""
    ls = {2: 1.0, 1: -1.0, 0: -2.0}
    f = {2: 1 / 6, 1: -5 / 6, 0: 5 / 3}
    return {j: e_ev + zeta_s_ev * ls[j] + rho_ev * f[j] for j in (0, 1, 2)}


def fit_triplet_p_intervals(e01: float, e12: float) -> tuple[float, float]:
    """Solve (zeta_S, rho) from the two measured intervals E(J=0) - E(J=1) = e01 and E(J=1) - E(J=2) = e12 (same unit, any sign).
    E0 - E1 = -zeta + (5/2) rho and E1 - E2 = -2 zeta - rho: an exact 2 x 2 solve, so the fit has no spare check."""
    a = np.array([[-1.0, 2.5], [-2.0, -1.0]])
    zeta, rho = np.linalg.solve(a, np.array([e01, e12]))
    return float(zeta), float(rho)


def _j_projectors():
    """Projectors onto the J = 0, 1, 2 subspaces of the 12-state space (spectral projectors of J^2)."""
    o = _ops()[0]
    j2 = sum((o["L" + c] + o["S" + c]) @ (o["L" + c] + o["S" + c]) for c in "xyz")
    w, u = np.linalg.eigh(j2)
    proj = {}
    for jj in (0, 1, 2):
        cols = u[:, np.abs(w - jj * (jj + 1)) < 1e-8]
        proj[jj] = cols @ cols.conj().T
    return j2, proj


def classify(b_tesla: float, h: np.ndarray) -> pd.DataFrame:
    """Eigenstates of a 1s2p Hamiltonian, sorted by energy, with labels that stay honest at finite field.

    Columns: energy_ev; mj (<J_z>) and m_j (the exact integer: J_z commutes with H for B along z); singlet
    (S = 0 probability); j2 (<J^2>); p_j0, p_j1, p_j2 (weights of the state in the J = 0, 1, 2 subspaces);
    j (the DOMINANT J, i.e. the J with the largest weight, not the rounded root of <J^2>); j_purity (that weight).

    At B = 0 every state has j_purity = 1. At finite field the Zeeman term couples J to J +- 1 at equal M_J, so
    J is NOT a good quantum number: use j only when j_purity is close to 1 (for the nominal 2^3P levels it is
    0.999 at 0.01 T, 0.90 at 0.1 T, and about 0.5 for some states above 1 T), and prefer m_j plus the weights
    p_j*, or the adiabatic parentage label of classify_tracked.
    """
    o = _ops()[0]
    s1s2 = _ops()[4]
    jz = o["Lz"] + o["Sz"]
    j2, proj = _j_projectors()
    psing = 0.25 * _I12 - s1s2
    hh = h + 1e-12 * jz if b_tesla == 0 else h         # lifts the M_J degeneracy at B = 0 only to get clean M_J states
    w, v = np.linalg.eigh(hh)
    ex = lambda m, k: float(np.real(v[:, k].conj() @ m @ v[:, k]))
    rows = []
    for k in range(12):
        pj = [ex(proj[jj], k) for jj in (0, 1, 2)]
        jmax = int(np.argmax(pj))
        mj = ex(jz, k)
        rows.append((w[k], mj, int(round(mj)), ex(psing, k), ex(j2, k), *pj, jmax, pj[jmax]))
    return pd.DataFrame(rows, columns=["energy_ev", "mj", "m_j", "singlet", "j2", "p_j0", "p_j1", "p_j2", "j", "j_purity"])


def classify_tracked(b_values, build) -> pd.DataFrame:
    """Follow every level from the lowest field to the highest by eigenvector overlap (solvers.track_from_builder) and
    attach its ZERO-FIELD PARENTAGE, e.g. "2^3P_1": the term and J of the state it connects to as B -> 0. This is the
    label that remains meaningful when J is no longer a good quantum number. Returns one row per (field, level) with
    b_tesla, level (tracked index), parent, m_j, energy_ev, singlet, j2, j_purity, p_j0..p_j2. The first value of
    b_values should be small compared with the 2^3P fine structure (about 0.2 T per 2.3 GHz), for example 1e-4 T."""
    from .solvers import track_from_builder
    b_values = np.asarray(b_values, dtype=float)
    e, v = track_from_builder(build, b_values, return_vectors=True)
    o = _ops()[0]
    jz = o["Lz"] + o["Sz"]
    j2, proj = _j_projectors()
    psing = 0.25 * _I12 - _ops()[4]
    ex = lambda m, vec: float(np.real(vec.conj() @ m @ vec))
    parent = {}
    for k in range(12):
        vec = v[0][:, k]
        jdom = int(np.argmax([ex(proj[jj], vec) for jj in (0, 1, 2)]))
        parent[k] = f"2^{1 if ex(psing, vec) > 0.5 else 3}P_{jdom}"
    rows = []
    for i, b in enumerate(b_values):
        for k in range(12):
            vec = v[i][:, k]
            pj = [ex(proj[jj], vec) for jj in (0, 1, 2)]
            rows.append((b, k, parent[k], int(round(ex(jz, vec))), e[i, k], ex(psing, vec), ex(j2, vec), max(pj), *pj))
    return pd.DataFrame(rows, columns=["b_tesla", "level", "parent", "m_j", "energy_ev", "singlet", "j2", "j_purity",
                                       "p_j0", "p_j1", "p_j2"])


# ---- Slater integrals from hydrogenic orbitals ---------------------------------------------------------------------------
def _radial(n: int, l: int, z: float, r: np.ndarray) -> np.ndarray:
    """Hydrogenic radial function R_nl(r) (r in bohr, effective charge z), normalised to integral R^2 r^2 dr = 1; n = 1, 2 only."""
    if (n, l) == (1, 0):
        return 2 * z ** 1.5 * np.exp(-z * r)
    if (n, l) == (2, 0):
        return 2 * (z / 2) ** 1.5 * (1 - z * r / 2) * np.exp(-z * r / 2)
    if (n, l) == (2, 1):
        return (z / 2) ** 1.5 * z * r / np.sqrt(3) * np.exp(-z * r / 2)
    raise ValueError("only 1s, 2s, 2p are implemented")


def slater_integral(k: int, a, b, c, d, n_grid: int = 200_000, r_max: float = 80.0) -> float:
    """R^k(ab; cd) = integral integral R_a(r1) R_c(r1) (r_<^k / r_>^{k+1}) R_b(r2) R_d(r2) r1^2 r2^2 dr1 dr2 in hartree,
    each orbital given as (n, l, z_eff). Electron 1 carries orbitals a and c, electron 2 carries b and d. The direct integral
    F^k(1s, nl) is R^k(1s nl; 1s nl) = slater_integral(k, a, b, a, b); the exchange integral G^k is slater_integral(k, a, b, b, a)."""
    r = np.linspace(1e-9, r_max, n_grid)
    ra, rc = _radial(*a[:2], a[2], r), _radial(*c[:2], c[2], r)
    rb, rd = _radial(*b[:2], b[2], r), _radial(*d[:2], d[2], r)
    rho1 = ra * rc * r ** 2                    # electron-1 density weight
    rho2 = rb * rd * r ** 2
    inner_low = cumulative_trapezoid(rho2 * r ** k, r, initial=0.0)                           # int_0^r rho2 r2^k
    inner_high = -cumulative_trapezoid((rho2 / r ** (k + 1))[::-1], r[::-1], initial=0.0)[::-1]   # int_r^inf rho2 / r2^(k+1)
    pot = inner_low / r ** (k + 1) + inner_high * r ** k
    return float(np.trapezoid(rho1 * pot, r))


def exchange_g_hartree(k: int, nl_up: tuple[int, int], z_1s: float = 2.0, z_nl: float = 1.0) -> float:
    """Hydrogenic estimate of G^k(1s, nl): R^k(1s nl; nl 1s). z_1s and z_nl are effective charges (a model input)."""
    a, b = (1, 0, z_1s), (nl_up[0], nl_up[1], z_nl)
    return slater_integral(k, a, b, b, a)


def direct_f_hartree(k: int, nl_up: tuple[int, int], z_1s: float = 2.0, z_nl: float = 1.0) -> float:
    """Hydrogenic estimate of F^k(1s, nl) = R^k(1s nl; 1s nl)."""
    a, b = (1, 0, z_1s), (nl_up[0], nl_up[1], z_nl)
    return slater_integral(k, a, b, a, b)
