r"""Hydrogen Balmer lines in a magnetic field: H-alpha (n = 3 -> 2), and other n -> 2 lines.

Unlike hydrogen.py (levels only, n = 2), this module produces a SPECTRUM: light is emitted when an atom
jumps from an upper manifold (n = 3 has 18 states with spin) to a lower one (n = 2 has 8 states).

Model
-----
Each manifold is block-diagonal in l:   H = sum over l of  [ E_l (+) spin-orbit ] + Zeeman.
* Fine structure: the Dirac energy E_nj (leading order in alpha) is imposed on every (n, l, j) level by
  choosing, for each l, a constant a_l and a spin-orbit constant zeta_l such that a_l + zeta_l L.S has the
  two Dirac eigenvalues. So 3p3/2 and 3d3/2 come out exactly degenerate, as in Dirac theory.
* Lamb shift: ns levels are raised by 1057.864 MHz * 8/n^3 (measured for n = 2, scaled as 1/n^3 for others;
  an approximation, accurate to a few percent, and tiny next to the Zeeman splitting at the fields used).
* Zeeman: mu_B B (L_z + g_s S_z), as in the rest of the project.
* Transition strength: |<u| D_q |k>|^2 between exact eigenstates, with D_q = R_{n'l',nl} * (angular part
  from the Wigner-Eckart theorem, transitions.dipole_matrices). The radial integrals R are computed
  numerically from the hydrogen wavefunctions.
Not modelled: hyperfine structure, the diamagnetic term (about 0.02 cm^-1 at 20 T for n = 3, negligible
here), and the Lamb shift of l > 0 levels. The output has the same columns as
optics.spectrometer.sodium_d_components, so every polariser, analyser and plot of the optics package works.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import constants as sc
from scipy.integrate import quad
from scipy.linalg import block_diag
from scipy.special import eval_genlaguerre, factorial

from .constants import HC_EV_CM, ev_to_wavenumber, wavenumber_to_ev
from .hamiltonians import coupled_ops, h_zeeman
from .optics.spectrometer import vacuum_cm1_to_air_nm
from .transitions import POLARISATION, dipole_matrices

ALPHA = sc.alpha
RY_INF_CM1 = sc.physical_constants["Rydberg constant"][0] / 100.0          # infinite nuclear mass
RY_H_CM1 = RY_INF_CM1 / (1 + sc.m_e / sc.m_p)                              # hydrogen (reduced mass)
LAMB_2S_MHZ = 1057.864                                                     # measured, E(2S1/2) - E(2P1/2)
MHZ_TO_CM1 = 1e6 / (sc.c * 100.0)
RATE_AU_PER_S = 1.0 / sc.physical_constants["atomic unit of time"][0]
LINE_NAMES = {3: "Halpha", 4: "Hbeta", 5: "Hgamma"}


# ------------------------------------------------------------------ radial part

def radial_function(n: int, l: int, r):
    """Hydrogen radial function R_nl(r) in atomic units (r in Bohr radii), normalised to int R^2 r^2 dr = 1."""
    rho = 2.0 * np.asarray(r, dtype=float) / n
    norm = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho ** l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


@lru_cache(maxsize=None)
def radial_integral(n_up: int, l_up: int, n_low: int, l_low: int) -> float:
    """R = int R_{n'l'}(r) r R_{nl}(r) r^2 dr in Bohr radii (the radial factor of the dipole matrix element)."""
    f = lambda r: float(radial_function(n_up, l_up, r) * radial_function(n_low, l_low, r) * r ** 3)
    return float(quad(f, 0.0, 70.0 * max(n_up, n_low) ** 2, limit=400)[0])


def einstein_a(n_up: int, l_up: int, n_low: int, l_low: int) -> float:
    """Spontaneous decay rate (1/s) of the level n_up,l_up to n_low,l_low (Bethe-Salpeter formula):
    A = (4/3) alpha^3 w^3 * l_max / (2 l_up + 1) * R^2, with w the Bohr transition energy in hartree."""
    if abs(l_up - l_low) != 1:
        return 0.0
    w = RY_H_CM1 * (1 / n_low ** 2 - 1 / n_up ** 2) / (2 * RY_INF_CM1)
    r = radial_integral(n_up, l_up, n_low, l_low)
    return float(4 / 3 * ALPHA ** 3 * w ** 3 * max(l_up, l_low) / (2 * l_up + 1) * r ** 2 * RATE_AU_PER_S)


# ------------------------------------------------------------------ energies

def dirac_energy_cm1(n: int, j: float) -> float:
    """Dirac energy of level (n, j) in cm^-1, to first order in alpha^2 (rest energy removed)."""
    return -RY_H_CM1 / n ** 2 - RY_H_CM1 * ALPHA ** 2 / n ** 3 * (1 / (j + 0.5) - 3 / (4 * n))


def lamb_shift_cm1(n: int) -> float:
    """Lamb shift of an ns level above the Dirac value (measured at n = 2, scaled as 1/n^3)."""
    return LAMB_2S_MHZ * 8.0 / n ** 3 * MHZ_TO_CM1


def _ls(l: int) -> np.ndarray:
    o = coupled_ops(l)
    return o["Lx"] @ o["Sx"] + o["Ly"] @ o["Sy"] + o["Lz"] @ o["Sz"]


def zero_field_block_ev(n: int, l: int, lamb: bool = True) -> np.ndarray:
    """H_0 + spin-orbit block of one l, with the two Dirac energies, in eV."""
    dim = 2 * (2 * l + 1)
    if l == 0:
        e = dirac_energy_cm1(n, 0.5) + (lamb_shift_cm1(n) if lamb else 0.0)
        return wavenumber_to_ev(e) * np.eye(dim, dtype=complex)
    e_plus, e_minus = dirac_energy_cm1(n, l + 0.5), dirac_energy_cm1(n, l - 0.5)
    zeta = (e_plus - e_minus) / (l + 0.5)                   # E(l+1/2) - E(l-1/2) = zeta (l + 1/2)
    a = e_plus - zeta * l / 2                               # eigenvalues of L.S are l/2 and -(l+1)/2
    return wavenumber_to_ev(a) * np.eye(dim, dtype=complex) + wavenumber_to_ev(zeta) * _ls(l)


def manifold_hamiltonian(n: int, b_tesla: float, lamb: bool = True) -> np.ndarray:
    """Hamiltonian (eV) of all states of principal quantum number n, basis ordered l = 0, 1, ..., n-1."""
    return block_diag(*[zero_field_block_ev(n, l, lamb) + h_zeeman(l, b_tesla) for l in range(n)])


# ------------------------------------------------------------------ transitions

def full_dipole(n_up: int, n_low: int) -> dict[int, np.ndarray]:
    """Dipole matrices D_q (dim_up x dim_low) between the two manifolds, radial factors included."""
    dim = lambda n: [2 * (2 * l + 1) for l in range(n)]
    du, dl = dim(n_up), dim(n_low)
    ou, ol = np.concatenate([[0], np.cumsum(du)]), np.concatenate([[0], np.cumsum(dl)])
    out = {q: np.zeros((ou[-1], ol[-1])) for q in (-1, 0, 1)}
    for lu in range(n_up):
        for ll in range(n_low):
            if abs(lu - ll) != 1:
                continue
            r = radial_integral(n_up, lu, n_low, ll)
            ang = dipole_matrices(lu, ll)
            for q in (-1, 0, 1):
                out[q][ou[lu]:ou[lu + 1], ol[ll]:ol[ll + 1]] = r * ang[q]
    return out


def balmer_components(b_tesla: float, n_up: int = 3, n_low: int = 2, lamb: bool = True,
                      threshold: float = 1e-7) -> pd.DataFrame:
    """Zeeman components of the n_up -> n_low line (default H-alpha) at field B.

    Columns (same as optics.spectrometer.sodium_d_components): line, q, polarisation, wavelength_nm (air),
    shift_pm (from the unperturbed Bohr line, fine structure and Zeeman included), rate (relative emission
    rate = squared dipole matrix element in Bohr radii squared). B must be above zero.
    """
    if b_tesla <= 0:
        raise ValueError("b_tesla must be positive")
    eu, vu = np.linalg.eigh(manifold_hamiltonian(n_up, b_tesla, lamb))
    el, vl = np.linalg.eigh(manifold_hamiltonian(n_low, b_tesla, lamb))
    amps = {q: vu.conj().T @ d @ vl for q, d in full_dipole(n_up, n_low).items()}
    top = max((np.abs(a) ** 2).max() for a in amps.values())
    sigma0 = RY_H_CM1 * (1 / n_low ** 2 - 1 / n_up ** 2)
    lam0 = float(vacuum_cm1_to_air_nm(sigma0))
    name = LINE_NAMES.get(n_up, f"{n_up}-{n_low}") if n_low == 2 else f"{n_up}-{n_low}"
    rows = []
    for q, a in amps.items():
        rate = np.abs(a) ** 2
        for i, k in zip(*np.where(rate > threshold * top)):
            sigma = ev_to_wavenumber(eu[i] - el[k])
            lam = float(vacuum_cm1_to_air_nm(sigma))
            rows.append((name, q, POLARISATION[q], lam, (lam - lam0) * 1e3, float(rate[i, k])))
    df = pd.DataFrame(rows, columns=["line", "q", "polarisation", "wavelength_nm", "shift_pm", "rate"])
    return df.sort_values("wavelength_nm").reset_index(drop=True)


def doppler_fwhm_pm(wavelength_nm: float = 656.28, temperature_k: float = 300.0, mass_amu: float = 1.00794) -> float:
    """Thermal Doppler full width at half maximum (pm): lambda * sqrt(8 k T ln2 / (m c^2))."""
    m = mass_amu * sc.atomic_mass
    return float(wavelength_nm * np.sqrt(8 * sc.k * temperature_k * np.log(2) / (m * sc.c ** 2)) * 1e3)
