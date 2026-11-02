"""Matplotlib plots and animations of Zeeman level splitting."""
from __future__ import annotations

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .constants import MU_B_EV
from .solvers import track_levels


def _setup(l, b_max, zeta_ev, n_points):
    b = np.linspace(0, b_max, n_points)
    e = track_levels(l, b, zeta_ev)
    return b, e


def plot_energy_levels(l: int, b_max: float, zeta_ev: float, n_points: int = 400, ax=None, show=False):
    """Static energy-vs-B diagram (weak field -> Paschen-Back)."""
    b, e = _setup(l, b_max, zeta_ev, n_points)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for k in range(e.shape[1]):
        ax.plot(b, e[:, k] * 1e3, lw=1.6)
    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Energy (meV)")
    ax.set_title(f"Zeeman splitting, l = {l} (spin-orbit zeta = {zeta_ev*1e3:.3f} meV)")
    ax.grid(alpha=0.3)
    if show:
        plt.show()
    return ax


def animate_splitting(l: int, b_max: float, zeta_ev: float, n_frames: int = 120, save_path: str | None = None):
    """Animate the levels as B grows; the traced curves show the transition to Paschen-Back."""
    b, e = _setup(l, b_max, zeta_ev, n_frames)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, b_max)
    ax.set_ylim(e.min() * 1e3 * 1.1, e.max() * 1e3 * 1.1)
    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Energy (meV)")
    ax.grid(alpha=0.3)
    lines = [ax.plot([], [], lw=1.6)[0] for _ in range(e.shape[1])]
    dots = [ax.plot([], [], "o", ms=5, color=ln.get_color())[0] for ln in lines]

    def update(i):
        for k, (ln, dt) in enumerate(zip(lines, dots)):
            ln.set_data(b[: i + 1], e[: i + 1, k] * 1e3)
            dt.set_data([b[i]], [e[i, k] * 1e3])
        return lines + dots

    anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow" if save_path.endswith(".gif") else None, fps=25)
    return anim


POL_COLORS = {"pi": "tab:blue", "sigma+": "tab:red", "sigma-": "tab:green"}


def plot_stick_spectrum(df, centre_cm1: float, ax=None, title: str = ""):
    """Stick spectrum of Zeeman components: x = shift from the zero-field line, height = intensity.

    df is the output of transitions.zeeman_lines (already restricted to one spectral line).
    Colours show the polarisation (pi = blue, sigma = red/green).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    for pol, color in POL_COLORS.items():
        sub = df[df["polarisation"] == pol]
        ax.vlines(sub["wavenumber_cm1"] - centre_cm1, 0, sub["intensity"], color=color, lw=3, label=pol)
    ax.set_xlabel(r"Shift from zero-field line (cm$^{-1}$)")
    ax.set_ylabel("Relative intensity")
    ax.set_title(title)
    ax.set_ylim(0, 1.35 * df["intensity"].max())
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)
    return ax


TERM_COLORS = {"2S1/2": "tab:red", "2P1/2": "tab:blue", "2P3/2": "tab:green"}


def plot_n2_levels(b, energies_ev, labels, ax=None, mark_fields_gauss=(), title: str = ""):
    """Hydrogen n = 2 levels versus B. Colour = term, energy in GHz above 2P1/2 at zero field.

    b, energies_ev, labels come from hydrogen.track_n2. mark_fields_gauss draws dotted vertical lines.
    """
    from .hydrogen import ev_to_mhz

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5))
    seen = set()
    for _, row in labels.iterrows():
        name = row.term if row.term not in seen else None
        seen.add(row.term)
        ax.plot(b, ev_to_mhz(energies_ev[:, int(row.level)]) / 1e3, color=TERM_COLORS[row.term], lw=1.5, label=name)
    for g in mark_fields_gauss:
        ax.axvline(g * 1e-4, color="k", ls=":", lw=1)
    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Energy above 2P1/2 at B = 0 (GHz)")
    ax.set_title(title or "Hydrogen n = 2 (fine structure + Lamb shift)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def b_scale_tesla(zeta_ev: float) -> float:
    """Field at which mu_B*B equals zeta: marks the weak-to-strong field crossover."""
    return zeta_ev / MU_B_EV
