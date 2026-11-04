"""Simulated Fabry-Perot ring pattern: the instrument that can actually resolve the Zeeman components.

A Fabry-Perot etalon (gap d, mirror reflectivity R) sends light of wavelength lambda into rings at angles theta where
2 d cos(theta) = m lambda. The transmission is the Airy function
    T = 1 / (1 + (4 R / (1 - R)^2) sin^2(delta / 2)),   delta = 4 pi d cos(theta) / lambda.
Each Zeeman component has its own wavelength, so each makes its own ring; close components make a ring that splits.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..animation import POL_COLORS
from .polarization import linear_polariser
from .spectrometer import observe, sodium_d_components

POL_LABEL = {"pi": "π", "sigma+": "σ+", "sigma-": "σ−"}


def reflectivity_for_finesse(finesse: float) -> float:
    """Mirror reflectivity R giving a reflectivity finesse F = pi sqrt(R) / (1 - R) (solved exactly)."""
    c = np.pi / finesse
    # F(1-R) = pi sqrt(R): with x = sqrt(R):  F x^2 + pi x - F = 0
    x = (-np.pi + np.sqrt(np.pi ** 2 + 4 * finesse ** 2)) / (2 * finesse)
    return float(x ** 2) if c > 0 else 0.0


def airy(theta_rad, wavelength_nm, gap_mm: float, reflectivity: float):
    """Airy transmission for light of one wavelength at angle theta (rad)."""
    delta = 4 * np.pi * gap_mm * 1e6 * np.cos(theta_rad) / wavelength_nm
    coef = 4 * reflectivity / (1 - reflectivity) ** 2
    return 1.0 / (1.0 + coef * np.sin(delta / 2) ** 2)


def radial_profile(theta_rad, df, gap_mm, reflectivity, weight_col="observed"):
    """Sum of the Airy patterns of all components (incoherent), weighted by observed intensity."""
    total = np.zeros_like(theta_rad, dtype=float)
    for r in df.itertuples():
        total += getattr(r, weight_col) * airy(theta_rad, r.wavelength_nm, gap_mm, reflectivity)
    return total


def ring_angle(n: int, gap_mm: float, wavelength_nm: float = 588.995) -> float:
    """Angle (rad) of the n-th bright ring (n = 0 is the central spot, or the nearest ring to it) of one wavelength."""
    m_axis = 2 * gap_mm * 1e6 / wavelength_nm
    return float(np.arccos((np.floor(m_axis) - n) / m_axis))


def _d2(b_tesla):
    c = sodium_d_components(b_tesla)
    return c[c.line == "D2"].reset_index(drop=True)


def fig_fabry_perot_rings(b_values=(0.0005, 0.15, 0.3), gap_mm: float = 5.0, finesse: float = 30.0, n_pix: int = 500):
    """Rows: no filter and polariser perpendicular to B. Columns: field. Each panel is a ring image of one
    interference order of the sodium D2 line (the D1 line lies in a different order and is filtered out)."""
    refl = reflectivity_for_finesse(finesse)
    fig, axes = plt.subplots(2, len(b_values), figsize=(3.6 * len(b_values), 7.2))
    theta_max = 0.5 * (ring_angle(2, gap_mm) + ring_angle(3, gap_mm))      # shows rings 0 to 2
    xs = np.linspace(-1, 1, n_pix)
    xx, yy = np.meshgrid(xs, xs)
    theta = np.sqrt(xx ** 2 + yy ** 2) * theta_max
    for j, b in enumerate(b_values):
        d2 = _d2(max(b, 1e-3))
        for i, (name, m) in enumerate([("no filter", None), ("polariser ⊥ B", linear_polariser(90.0))]):
            d = observe(d2, 90.0, m)
            d = d[d.observed > 1e-3 * d.observed.max()]
            img = radial_profile(theta, d, gap_mm, refl)
            ax = axes[i, j]
            ax.imshow(img / img.max(), cmap="afmhot", extent=(-1, 1, -1, 1), vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"B = {b:g} T" if i == 0 else "")
            if j == 0:
                ax.set_ylabel(name)
    fig.suptitle(f"Fabry-Perot rings, sodium D2 (gap {gap_mm:g} mm, finesse {finesse:g}); each ring splits into its Zeeman components as B grows")
    fig.tight_layout()
    return fig


def fig_fabry_perot_profile(b_tesla: float = 0.3, gap_mm: float = 5.0, finesse: float = 30.0):
    """Ring brightness against angle at one field: the components resolved as separate peaks, coloured by polarisation."""
    refl = reflectivity_for_finesse(finesse)
    d2 = _d2(b_tesla)
    th0 = ring_angle(2, gap_mm)
    th = np.linspace(th0 - 1.5e-3, th0 + 1.5e-3, 4000)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, (name, m) in zip(axes, [("no filter", None), ("polariser ⊥ B (σ only)", linear_polariser(90.0)),]):
        d = observe(d2, 90.0, m)
        d = d[d.observed > 1e-3 * d.observed.max()]
        ax.plot(th * 1e3, radial_profile(th, d, gap_mm, refl), "k", lw=1)
        for pol, col in POL_COLORS.items():
            s = d[d.polarisation == pol]
            if len(s):
                prof = radial_profile(th, s, gap_mm, refl)
                ax.fill_between(th * 1e3, prof, color=col, alpha=0.35, label=POL_LABEL[pol])
        ax.set_ylabel(name); ax.legend(fontsize=8, loc="upper right")
    axes[1].set_xlabel("angle from the optical axis (mrad)")
    axes[0].set_title(f"Sodium D2 at {b_tesla:g} T through a Fabry-Perot (gap {gap_mm:g} mm, finesse {finesse:g})")
    fig.tight_layout()
    return fig
