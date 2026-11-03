"""Figures for the hydrogen H-alpha Zeeman pattern (n = 3 -> 2, 656 nm).

fig_halpha_regimes: the spectrum at weak, intermediate and strong field, seen through no filter, a polariser
                    parallel to B and a polariser perpendicular to B (rows), blurred by thermal Doppler width.
fig_halpha_fan:     component wavelengths against B (dot area proportional to emission rate).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..animation import POL_COLORS
from ..hydrogen_spectrum import balmer_components, doppler_fwhm_pm
from .polarization import linear_polariser
from .spectrometer import gaussian_spectrum, observe

POL_LABEL = {"pi": "π", "sigma+": "σ+", "sigma-": "σ−"}
FILTERS = [("no filter", None), ("polariser ∥ B", linear_polariser(0.0)), ("polariser ⊥ B", linear_polariser(90.0))]


def _visible(df, col, frac=0.003):
    return df[df[col] > frac * df[col].max()]


def fig_halpha_regimes(b_values=(0.05, 1.0, 6.0), temperature_k: float = 300.0):
    """Rows: filters. Columns: field strengths. Sticks coloured by polarisation, grey curve = Doppler-blurred."""
    fwhm_pm = doppler_fwhm_pm(temperature_k=temperature_k)
    fig, axes = plt.subplots(3, len(b_values), figsize=(4.2 * len(b_values), 7.2), sharey="row")
    for j, b in enumerate(b_values):
        comps = balmer_components(b)
        for i, (name, m) in enumerate(FILTERS):
            ax = axes[i, j]
            df = _visible(observe(comps, 90.0, m), "observed")
            half = max(df.shift_pm.abs().max() * 1.15, 3 * fwhm_pm)
            grid = np.linspace(-half, half, 1500)
            blur = gaussian_spectrum(grid, df.shift_pm.to_numpy(), df.observed.to_numpy(), fwhm_pm)
            top = df.observed.max()
            ax.fill_between(grid, blur / blur.max() * top, color="0.85", lw=0)
            for pol, col in POL_COLORS.items():
                s = df[df.polarisation == pol]
                ax.vlines(s.shift_pm, 0, s.observed, color=col, lw=2.2, label=POL_LABEL[pol])
            ax.set_xlim(-half, half)
            ax.set_ylim(0, top * 1.1)
            if i == 0:
                ax.set_title(f"B = {b:g} T")
            if j == 0:
                ax.set_ylabel(f"{name}\nrelative intensity")
            if i == 2:
                ax.set_xlabel("shift from the Bohr line (pm)")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=8, title="polarisation")
    fig.suptitle(f"Hα (656 nm) seen across the field; grey = {temperature_k:g} K Doppler width ({fwhm_pm:.1f} pm FWHM)")
    fig.tight_layout()
    return fig


def fig_halpha_fan(b_max: float = 8.0, n_points: int = 60):
    """Component positions versus field. Dashed lines: the normal Zeeman shift +-20.11 pm/T around the centre."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for b in np.linspace(b_max / n_points, b_max, n_points):
        df = _visible(balmer_components(b), "rate", 0.02)
        for pol, col in POL_COLORS.items():
            s = df[df.polarisation == pol]
            ax.scatter(np.full(len(s), b), s.shift_pm, s=4 + 90 * s.rate / df.rate.max(), color=col, alpha=0.7, lw=0)
    for pol, col in POL_COLORS.items():
        ax.scatter([], [], color=col, label=POL_LABEL[pol])
    bb = np.array([0, b_max])
    ax.plot(bb, 20.11 * bb, "k--", lw=0.8, label="normal Zeeman ±20.11 pm/T")
    ax.plot(bb, -20.11 * bb, "k--", lw=0.8)
    ax.set_xlabel("magnetic field B (T)")
    ax.set_ylabel("shift from the Bohr line (pm)")
    ax.set_title("Hα components versus field (dot area ∝ emission rate)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def animate_halpha_field_ramp(save_path=None, b_max: float = 8.0, frames: int = 60, temperature_k: float = 300.0):
    """GIF: B rises from near zero to b_max; three panels (no filter, polariser parallel / perpendicular to B).
    Sticks are coloured by polarisation, the grey curve is the Doppler-blurred line."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    fwhm = doppler_fwhm_pm(temperature_k=temperature_k)
    bs = np.linspace(0.02, b_max, frames)
    xlim = 20.11 * b_max * 1.2
    grid = np.linspace(-xlim, xlim, 1600)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.4), sharex=True)
    ymax = 6.5

    def draw(k):
        b = bs[k]
        comps = balmer_components(b)
        for ax, (name, m) in zip(axes, FILTERS):
            ax.clear()
            d = _visible(observe(comps, 90.0, m), "observed", 0.003)
            blur = gaussian_spectrum(grid, d.shift_pm.to_numpy(), d.observed.to_numpy(), fwhm)
            ax.fill_between(grid, blur / max(blur.max(), 1e-12) * min(d.observed.max(), ymax), color="0.85", lw=0)
            for pol, col in POL_COLORS.items():
                s = d[d.polarisation == pol]
                ax.vlines(s.shift_pm, 0, s.observed, color=col, lw=2.2, label=POL_LABEL[pol])
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(0, ymax)
            ax.set_ylabel(name)
        axes[0].set_title(f"Hα in a magnetic field: B = {b:.2f} T   (grey: {temperature_k:g} K Doppler width, {fwhm:.1f} pm)")
        axes[0].legend(loc="upper left", fontsize=8, ncol=3)
        axes[2].set_xlabel("shift from the Bohr line (pm)")
        return []

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.08)
    anim = FuncAnimation(fig, draw, frames=frames, interval=90, blit=False)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=11))
    return anim
