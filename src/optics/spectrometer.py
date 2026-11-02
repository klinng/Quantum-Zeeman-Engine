r"""The virtual spectrometer: a lamp, a filter, and the Zeeman pattern of sodium seen through optics.

Chain:  source in field B  ->  line of sight (angle theta from B)  ->  polariser / analyser
        ->  prism (colours)  and/or  high-resolution stage (Zeeman components)

Only the sodium D lines have a Zeeman structure computed from the atomic model (src.transitions).
The other lamp lines are given only as colours and have illustrative relative intensities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import G_S, MU_B_EV, ev_to_wavenumber, wavenumber_to_ev
from ..transitions import zeeman_lines
from ..validation import zeta_from_fine_structure
from .polarization import component_jones

N_AIR = 1.000277            # refractive index of air near 589 nm (vacuum -> air wavelength)

# name, element, wavelength in air (nm), illustrative relative intensity
_LAMP = [
    ("Hg 404.7", "Hg", 404.66, 0.40), ("H-delta", "H", 410.17, 0.25), ("H-gamma", "H", 434.05, 0.45),
    ("Hg 435.8", "Hg", 435.83, 0.90), ("H-beta", "H", 486.13, 0.70), ("Hg 546.1", "Hg", 546.07, 1.00),
    ("Hg 577.0", "Hg", 576.96, 0.50), ("Hg 579.1", "Hg", 579.07, 0.55), ("Na D2", "Na", 588.995, 0.90),
    ("Na D1", "Na", 589.592, 0.60), ("H-alpha", "H", 656.28, 1.00),
]
LAMP_LINES = pd.DataFrame(_LAMP, columns=["name", "element", "wavelength_nm", "intensity"])


def vacuum_cm1_to_air_nm(sigma_cm1):
    """Vacuum wavenumber (cm^-1) to air wavelength (nm)."""
    return 1e7 / np.asarray(sigma_cm1, dtype=float) / N_AIR


def gaussian_spectrum(wl_grid_nm, centres_nm, weights, fwhm_nm: float) -> np.ndarray:
    """Sum of Gaussians (instrument profile) evaluated on a wavelength grid."""
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    wl = np.asarray(wl_grid_nm)[:, None]
    return (np.asarray(weights)[None, :] * np.exp(-0.5 * ((wl - np.asarray(centres_nm)[None, :]) / sigma) ** 2)).sum(axis=1)


def bandpass_transmission(wl_nm, centre_nm: float, fwhm_nm: float):
    """Gaussian band-pass colour filter: transmission 1 at the centre, 0.5 at +/- FWHM/2."""
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((np.asarray(wl_nm) - centre_nm) / sigma) ** 2)


def sodium_d_components(b_tesla: float, g_s: float = G_S) -> pd.DataFrame:
    """Zeeman components of the sodium D1 and D2 lines at field B (weak-to-moderate field).

    Columns: line, q, polarisation, wavelength_nm (air), shift_pm (from the zero-field line),
    rate (relative emission rate, every upper state emits 1 in total).
    B must be > 0 (at exactly 0 degenerate eigenvectors make individual rates meaningless).
    """
    if b_tesla <= 0:
        raise ValueError("b_tesla must be positive")
    zeta = zeta_from_fine_structure(17.196, l=1)
    gap = wavenumber_to_ev(16956.17) + zeta                    # puts 2P1/2 at the measured D1 energy
    centres_ev = {"D1": gap - zeta, "D2": gap + zeta / 2}
    mid = 0.5 * (centres_ev["D1"] + centres_ev["D2"])
    df = zeeman_lines(1, 0, b_tesla, zeta, 0.0, gap, g_s=g_s)
    df["line"] = np.where(df.energy_ev < mid, "D1", "D2")
    df["wavelength_nm"] = vacuum_cm1_to_air_nm(df.wavenumber_cm1)
    centre_nm = df.line.map({k: float(vacuum_cm1_to_air_nm(ev_to_wavenumber(v))) for k, v in centres_ev.items()})
    df["shift_pm"] = (df.wavelength_nm - centre_nm) * 1e3
    df = df.rename(columns={"intensity": "rate"})
    return df[["line", "q", "polarisation", "wavelength_nm", "shift_pm", "rate"]].reset_index(drop=True)


def observe(components: pd.DataFrame, theta_deg: float = 90.0, element: np.ndarray | None = None) -> pd.DataFrame:
    """Add the observed intensity of each component for a line of sight at theta from B.

    element is an optional 2x2 Jones matrix placed in the beam (polariser, analyser, wave plate
    combinations); None means no optical element. Observed intensity = rate * |M J(q, theta)|^2.
    """
    m = np.eye(2) if element is None else element
    th = np.radians(theta_deg)
    obs = [float(r.rate) * float(np.linalg.norm(m @ component_jones(int(r.q), th)) ** 2) for r in components.itertuples()]
    return components.assign(observed=obs)


def required_resolving_power(components: pd.DataFrame) -> float:
    """Resolving power lambda / d(lambda) needed to separate the closest pair of components in a line."""
    worst = np.inf
    for _, grp in components.groupby("line"):
        wl = np.sort(grp.wavelength_nm.to_numpy())
        gaps = np.diff(wl)
        gaps = gaps[gaps > 1e-9]
        if gaps.size:
            worst = min(worst, float(gaps.min()) / float(wl.mean()))
    return 1.0 / worst


def field_for_resolving_power(r: float, line_nm: float = 589.3) -> float:
    """Field (tesla) at which the closest components are just resolved by an instrument of power r.

    Uses the weak-field spacing (2/3) mu_B B between neighbouring sodium D components.
    """
    delta_sigma_cm1 = r_to_dsigma(r, line_nm)
    return delta_sigma_cm1 / ((2 / 3) * MU_B_EV / (wavenumber_to_ev(1.0)))


def r_to_dsigma(r: float, line_nm: float) -> float:
    """Smallest resolvable wavenumber difference (cm^-1) of an instrument with resolving power r."""
    return (1e7 / line_nm) / r
