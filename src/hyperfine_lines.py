r"""Sodium D lines with hyperfine structure: line positions and strengths in a magnetic field.

The electric-dipole operator acts on the electron only, so in the enlarged basis it is D (x) 1_I.
Line strength between eigenstates u (3P, 24 states) and l (3S, 8 states) is |<u| D_q (x) 1_I |l>|^2.
Frequencies are measured from the hyperfine-free D1 line (3P1/2 centre of gravity to 3S1/2 centre of gravity)
and given in MHz. The 3P3/2 electric-quadrupole constant B = 2.724 MHz is included (hyperfine.NA_B_MHZ). The F, F' labels of the zero-field lines are exact; in a field use m_F and the weights of hyperfine.f_weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import constants as sc
import matplotlib.pyplot as plt

from .hydrogen import ev_to_mhz
from .hyperfine import NA_I, h_na_3p, h_na_3s
from .transitions import POLARISATION, dipole_matrices
from .validation import zeta_from_fine_structure

FINE_NA_CM1 = 17.196          # NIST Handbook: 16973.368 - 16956.172
NA_MASS_KG = 22.98976928 * sc.atomic_mass


def sodium_d_hyperfine_lines(b_tesla: float, threshold: float = 1e-9) -> pd.DataFrame:
    """All hyperfine-Zeeman components of the sodium D1 and D2 lines at field B (tesla; B = 0 is allowed).

    Columns: line (D1 or D2), q, polarisation, freq_mhz (from the hyperfine-free D1 line), strength.
    Strength is normalised so that a lower state emits 1 in total (D1 plus D2).
    """
    zeta = zeta_from_fine_structure(FINE_NA_CM1, l=1)
    eu, vu = np.linalg.eigh(h_na_3p(b_tesla, zeta))
    el, vl = np.linalg.eigh(h_na_3s(b_tesla))
    n_i = int(round(2 * NA_I + 1))
    mid_ev = 0.5 * (0.0 + 1.5 * zeta)                      # between D1 (0) and D2 (1.5 zeta)
    rows = []
    for q, d in dipole_matrices(1, 0).items():
        amp = vu.conj().T @ np.kron(d, np.eye(n_i)) @ vl
        for i, j in zip(*np.where(np.abs(amp) ** 2 > threshold)):
            e = eu[i] - el[j]
            rows.append(("D1" if e < mid_ev else "D2", q, POLARISATION[q], float(ev_to_mhz(e)), float(abs(amp[i, j]) ** 2)))
    df = pd.DataFrame(rows, columns=["line", "q", "polarisation", "freq_mhz", "strength"])
    total_per_lower = df.strength.sum() / len(el)
    df["strength"] = df.strength / total_per_lower
    return df.sort_values("freq_mhz").reset_index(drop=True)


def group_lines(df: pd.DataFrame, line: str, resolution_mhz: float = 0.01) -> pd.DataFrame:
    """Merge components of one line that sit at the same frequency (needed at B = 0, where levels are degenerate)."""
    d = df[df.line == line].copy()
    d["bin"] = np.round(d.freq_mhz / resolution_mhz).astype(int)
    g = d.groupby("bin").agg(freq_mhz=("freq_mhz", "mean"), strength=("strength", "sum")).reset_index(drop=True)
    return g.sort_values("freq_mhz").reset_index(drop=True)


def doppler_fwhm_mhz(temperature_k: float = 300.0, wavenumber_cm1: float = 16973.368) -> float:
    """Doppler full width at half maximum of a sodium D line (MHz)."""
    nu0 = wavenumber_cm1 * 100.0 * sc.c
    return nu0 * np.sqrt(8 * np.log(2) * sc.k * temperature_k / (NA_MASS_KG * sc.c ** 2)) * 1e-6


def broadened_spectrum(df: pd.DataFrame, line: str, fwhm_mhz: float, freq: np.ndarray) -> np.ndarray:
    """Sum of Gaussian profiles, one per component, area proportional to strength."""
    d = df[df.line == line]
    sig = fwhm_mhz / (2 * np.sqrt(2 * np.log(2)))
    return (d.strength.to_numpy()[:, None] * np.exp(-0.5 * ((freq[None, :] - d.freq_mhz.to_numpy()[:, None]) / sig) ** 2)).sum(0) \
        / (sig * np.sqrt(2 * np.pi))


def fig_na_d2_hyperfine(b_max_t: float = 0.3, n_b: int = 41):
    """Left: zero-field D2 hyperfine lines (sticks) and the same seen through Doppler broadening at 300 K.
    Right: the D2 components against field; marker area = strength, colour = polarisation."""
    colors = {"pi": "tab:green", "sigma+": "tab:red", "sigma-": "tab:blue"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    z = group_lines(sodium_d_hyperfine_lines(0.0), "D2")
    d2c = z.freq_mhz.to_numpy() @ z.strength.to_numpy() / z.strength.sum()
    a1.vlines(z.freq_mhz - d2c, 0, z.strength, color="k", lw=2)
    fw = doppler_fwhm_mhz()
    f = np.linspace(z.freq_mhz.min() - d2c - 1500, z.freq_mhz.max() - d2c + 1500, 800)
    prof = broadened_spectrum(sodium_d_hyperfine_lines(0.0).assign(freq_mhz=lambda d: d.freq_mhz - d2c), "D2", fw, f + 0 * f)
    a1.plot(f, prof / prof.max() * z.strength.max(), color="tab:orange", label=f"Doppler broadened, 300 K (FWHM {fw:.0f} MHz)")
    a1.set_xlabel("frequency from D2 centre of gravity (MHz)"); a1.set_ylabel("strength")
    a1.set_title("Na D2 at zero field: 6 hyperfine lines, two groups 1772 MHz apart", fontsize=10); a1.legend(fontsize=8)
    for b in np.linspace(1e-4, b_max_t, n_b):
        d = sodium_d_hyperfine_lines(b)
        d = d[(d.line == "D2")]
        for pol, c in colors.items():
            s = d[d.polarisation == pol]
            a2.scatter(np.full(len(s), b * 1e4), s.freq_mhz - d2c, s=s.strength * 70, c=c, alpha=0.6, lw=0,
                       label=pol if b < 2e-4 else None)
    a2.set_xlabel("B (gauss)"); a2.set_ylabel("frequency from zero-field D2 centre (MHz)")
    a2.set_title("D2 components vs field (hyperfine Paschen–Back near 630 G)", fontsize=10)
    a2.legend(fontsize=8, markerscale=0.5)
    fig.tight_layout()
    return fig
