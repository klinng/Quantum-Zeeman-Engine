"""Sodium D lines from weak field to Paschen-Back (up to 60 T).

At weak field the D1 and D2 lines each split into their own anomalous Zeeman pattern (4 + 6 components).
Above the crossover field B* = zeta / mu_B = 24.6 T the field overpowers the spin-orbit coupling, the doublet
collapses into the normal Zeeman triplet (pi in the middle, sigma either side), and "D1" and "D2" lose their meaning
because lines from both cross over. So shifts here are measured from the centre of gravity of the doublet, not from D1 or D2.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..animation import POL_COLORS
from .instruments import B_STAR_TESLA
from .polarization import linear_polariser
from .spectrometer import gaussian_spectrum, observe, sodium_d_components, vacuum_cm1_to_air_nm

POL_LABEL = {"pi": "π", "sigma+": "σ+", "sigma-": "σ−"}
FILTERS = [("no filter", None), ("polariser ∥ B", linear_polariser(0.0)), ("polariser ⊥ B", linear_polariser(90.0))]
D1_CM1, FINE_CM1 = 16956.17, 17.196
COG_NM = float(vacuum_cm1_to_air_nm((2 * D1_CM1 + 4 * (D1_CM1 + FINE_CM1)) / 6))   # weights 2J+1 of the 2P levels


def components_vs_cog(b_tesla: float):
    """Sodium D components with shift_pm measured from the centre of gravity of the doublet."""
    d = sodium_d_components(b_tesla)
    d["shift_pm"] = (d.wavelength_nm - COG_NM) * 1e3
    return d


def fig_sodium_paschen_back_fan(b_max: float = 60.0, n_points: int = 120):
    """Component positions against field; dot area is proportional to the emission rate. The vertical line marks B*."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for b in np.linspace(b_max / n_points, b_max, n_points):
        d = components_vs_cog(b)
        d = d[d.rate > 0.03]
        for pol, col in POL_COLORS.items():
            s = d[d.polarisation == pol]
            ax.scatter(np.full(len(s), b), s.shift_pm, s=5 + 28 * s.rate, color=col, alpha=0.75, lw=0)
    for pol, col in POL_COLORS.items():
        ax.scatter([], [], color=col, label=POL_LABEL[pol])
    ax.axvline(B_STAR_TESLA, color="k", ls=":", lw=1)
    ax.text(B_STAR_TESLA + 0.6, ax.get_ylim()[0] * 0.92, "B* = ζ/μ_B ≈ 24.6 T", fontsize=8)
    ax.set_xlabel("magnetic field B (T)")
    ax.set_ylabel("shift from the doublet's centre of gravity (pm)")
    ax.set_title("Sodium D lines: from two anomalous patterns to the normal Zeeman triplet (dot area ∝ rate)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def animate_sodium_paschen_back(save_path=None, b_max: float = 60.0, frames: int = 70):
    """GIF: B rises to b_max; three views (no filter, polariser parallel and perpendicular to B)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    bs = np.linspace(0.3, b_max, frames)
    xlim = 2100.0
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.4), sharex=True)

    def draw(k):
        b = bs[k]
        comps = components_vs_cog(b)
        for ax, (name, m) in zip(axes, FILTERS):
            ax.clear()
            d = observe(comps, 90.0, m)
            d = d[d.observed > 0.01]
            for pol, col in POL_COLORS.items():
                s = d[d.polarisation == pol]
                ax.vlines(s.shift_pm, 0, s.observed, color=col, lw=2.6, label=POL_LABEL[pol])
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(0, 1.3)
            ax.set_ylabel(name)
            ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        regime = "weak field (anomalous)" if b < 0.3 * B_STAR_TESLA else ("crossover" if b < 1.5 * B_STAR_TESLA else "Paschen-Back")
        axes[0].set_title(f"Sodium D lines: B = {b:.1f} T   ({regime}; B* = {B_STAR_TESLA} T)")
        axes[0].legend(loc="upper left", fontsize=8, ncol=3)
        axes[2].set_xlabel("shift from the doublet's centre of gravity (pm)")
        return []

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.08)
    anim = FuncAnimation(fig, draw, frames=frames, interval=90, blit=False)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=11))
    return anim
