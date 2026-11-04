r"""Hydrogen n = 2 in an electric AND a magnetic field: Stark mixing and the Lamb-Retherford anticrossing.

A magnetic field never mixes 2s with 2p (hydrogen.py: the levels simply cross). An electric field does, because the
operator e E.r connects states of opposite parity: <2s|z|2p, m=0> = -3 a0. Adding it to the 8x8 n = 2 Hamiltonian

    H = H_n2(B) + e (E_z z + E_x x)

gives two effects. With E parallel to B only states of the same m_j mix (mixing at B = 0 and the Stark fan). With E
perpendicular to B (the motional field v x B seen by an atom flying through a magnet is of this kind) the states
2S1/2(m_j = -1/2) and 2P1/2(m_j = +1/2) also mix: the 573 G crossing becomes an ANTICROSSING, the effect behind the
Lamb-Retherford experiment. The quenching of the metastable 2S state (its admixture of 2P decays at 6.3e8 per second)
is computed from the same eigenvectors.
Units: field E in V/cm, energies in MHz (mhz_to_ev from hydrogen.py converts).
The 2S-2P matrix elements come from hydrogen_spectrum.full_dipole (Wigner-Eckart angular part times the radial integral).
Not modelled: hyperfine structure (it shifts the exact positions of the anticrossings).
"""
from __future__ import annotations

import numpy as np
from scipy import constants as sc

from .hydrogen import h_n2, ev_to_mhz, mhz_to_ev
from .hydrogen_spectrum import einstein_a, full_dipole
from .solvers import track_from_builder

A0_M = sc.physical_constants["Bohr radius"][0]
EV_PER_A0_VCM = A0_M * 100.0                  # e * a0 * (1 V/cm), in eV: a0 [m] * 100 V/m per V/cm
MHZ_PER_A0_VCM = ev_to_mhz(EV_PER_A0_VCM)     # about 1.28 MHz per (V/cm) for one Bohr radius of dipole
A_2P_1S = einstein_a(2, 1, 1, 0)              # decay rate of 2P (to 1S), 1/s


def dipole_operators():
    """Position operators (x, z) of the n = 2 manifold in units of a0 (Hermitian 8x8, basis of hydrogen.h_n2)."""
    d = full_dipole(2, 2)
    return (d[-1] - d[1]) / np.sqrt(2), d[0]


def h_n2_stark(b_tesla: float, e_vcm: float, angle_deg: float = 90.0, **kw) -> np.ndarray:
    """H_n2(B) + e E (cos(angle) z + sin(angle) x): E at `angle_deg` from B (0 = parallel, 90 = perpendicular)."""
    x, z = dipole_operators()
    a = np.radians(angle_deg)
    v = EV_PER_A0_VCM * e_vcm * (np.cos(a) * z + np.sin(a) * x)
    return h_n2(b_tesla, **kw) + v


def track_stark(b_values, e_vcm: float, angle_deg: float = 90.0, **kw):
    """Level energies (eV) along a field sweep, followed by eigenvector overlap. Returns (energies, eigenvectors)."""
    b = np.array(b_values, dtype=float)
    b[0] = max(b[0], 1e-6)
    return track_from_builder(lambda t: h_n2_stark(t, e_vcm, angle_deg, **kw), b, return_vectors=True)


def s_weights(vectors) -> np.ndarray:
    """Probability of each eigenvector (columns) being 2s (first two basis states)."""
    return np.sum(np.abs(vectors[:2, :]) ** 2, axis=0)


def quenching_rate(b_tesla: float, e_vcm: float, angle_deg: float = 90.0) -> np.ndarray:
    """Decay rate (1/s) of each eigenstate (ascending energy): its 2p weight times the 2P decay rate.
    A pure 2S state has no E1 decay; field mixing gives it one. (Two-photon decay, 8 per second, is ignored.)"""
    _, v = np.linalg.eigh(h_n2_stark(b_tesla, e_vcm, angle_deg))
    return (1.0 - s_weights(v)) * A_2P_1S


def perturbative_s_shift_mhz(e_vcm: float, lamb_mhz: float = 1057.864, fine_mhz: float = 10969.04) -> float:
    """Second-order Stark shift of 2S1/2 (m_j = 1/2) for E parallel to B = 0, from the standard formulas.
    2S couples to 2P1/2 with 3 x^2 and to 2P3/2 with 6 x^2 (x = e a0 E); used as an independent check."""
    x2 = (MHZ_PER_A0_VCM * e_vcm) ** 2
    return 3 * x2 / lamb_mhz + 6 * x2 / (lamb_mhz - fine_mhz)


def anticrossing_gap_mhz(e_vcm: float, b_window_t=(0.0540, 0.0600), n: int = 241) -> tuple[float, float]:
    """Smallest gap (MHz) between neighbouring levels in the field window around the 573 G crossing, for E
    perpendicular to B, and the field (T) where it occurs. In that window the closest pair is always the
    2S1/2 (m_j = -1/2) and 2P1/2 (m_j = +1/2) states."""
    bs = np.linspace(*b_window_t, n)
    gaps = np.array([np.diff(ev_to_mhz(np.linalg.eigvalsh(h_n2_stark(b, e_vcm, 90.0)))).min() for b in bs])
    k = int(np.argmin(gaps))
    return float(gaps[k]), float(bs[k])


def lifetime_s(rate_quench_per_s: float, two_photon_per_s: float = 8.2) -> float:
    """Lifetime of a metastable-2S-like state: 1 / (quenching rate + the two-photon decay of pure 2S, about 8.2 /s)."""
    return 1.0 / (rate_quench_per_s + two_photon_per_s)
