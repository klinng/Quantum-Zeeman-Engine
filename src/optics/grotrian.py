r"""Grotrian (level-and-arrow) diagram of Zeeman transitions, static and animated.

Upper manifold (here sodium 3p) in a top panel, lower manifold (3s) in a bottom panel. Each level is a short horizontal
bar placed at its quantum number M = <J_z> = <L_z + S_z>, which is exact at every field strength (the field points along z,
so M is conserved) even when m_j and m_l, m_s stop being good labels. Energies are measured from each manifold's
centre of gravity, because the gap between the manifolds (16 956 cm^-1) is thousands of times larger than the Zeeman splitting
(a broken energy axis; the arrows run between the two panels).

An arrow is drawn only when the transition has a non-zero matrix element: the arrow joins upper state u to lower state k when
I_uk (the squared Wigner-Eckart dipole element between the exact eigenvectors) exceeds `min_fraction` of the strongest line.
Colour = polarisation (pi blue, sigma+ red, sigma- green), width = intensity, so lines fade out as B grows.
By construction M_u - M_k = q for every arrow (selection rule), which the tests check.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from ..animation import POL_COLORS
from ..constants import G_S, ev_to_wavenumber, wavenumber_to_ev
from ..hamiltonians import coupled_ops, h_total
from ..transitions import zeeman_lines
from ..validation import zeta_from_fine_structure

D1_CM1, FINE_CM1 = 16956.17, 17.196
POL_LABEL = {"pi": "π (q = 0)", "sigma+": "σ+ (q = +1)", "sigma-": "σ− (q = −1)"}
LW_PER_INTENSITY = 3.6


def _m_of_states(l: int, vectors: np.ndarray) -> np.ndarray:
    o = coupled_ops(l)
    jz = o["Lz"] + o["Sz"]
    return np.array([np.real(vectors[:, k].conj() @ jz @ vectors[:, k]) for k in range(vectors.shape[1])])


def grotrian_data(b_tesla: float, l_up: int = 1, l_low: int = 0, zeta_ev: float | None = None,
                  gap_ev: float | None = None, g_s: float = G_S, min_fraction: float = 1e-3) -> dict:
    """Levels, quantum numbers and drawable arrows at one field. Defaults are the sodium D lines.

    Returns a dict: e_up, m_up, e_low, m_low (energies in cm^-1 from each manifold's centre; M = <J_z>) and
    `arrows`, a DataFrame (upper, lower, q, polarisation, intensity) with the near-zero transitions removed.
    """
    if zeta_ev is None:
        zeta_ev = zeta_from_fine_structure(FINE_CM1, l=1)
    if gap_ev is None:
        gap_ev = wavenumber_to_ev(D1_CM1) + zeta_ev
    eu, vu = np.linalg.eigh(h_total(l_up, b_tesla, zeta_ev, e0_ev=gap_ev, g_s=g_s))
    el, vl = np.linalg.eigh(h_total(l_low, b_tesla, 0.0, g_s=g_s))
    lines = zeeman_lines(l_up, l_low, b_tesla, zeta_ev, 0.0, gap_ev, g_s=g_s)
    arrows = lines[lines.intensity > min_fraction * lines.intensity.max()].reset_index(drop=True)
    return {"e_up": ev_to_wavenumber(eu - gap_ev), "m_up": _m_of_states(l_up, vu),
            "e_low": ev_to_wavenumber(el), "m_low": _m_of_states(l_low, vl), "arrows": arrows, "b_tesla": b_tesla}


def _bar_positions(m: np.ndarray, e: np.ndarray, half_width: float = 0.17) -> list[tuple[float, float]]:
    """x-ranges of the bars: states that share the same M are shifted sideways so they do not overlap."""
    out = [(0.0, 0.0)] * len(m)
    for val in np.unique(np.round(m, 6)):
        idx = [k for k in range(len(m)) if abs(m[k] - val) < 1e-6]
        idx.sort(key=lambda k: e[k])
        n = len(idx)
        for rank, k in enumerate(idx):
            centre = val + (rank - (n - 1) / 2) * 2.3 * half_width
            out[k] = (centre - half_width, centre + half_width)
    return out


def draw_grotrian(ax_up, ax_low, data: dict, ylim_up=None, ylim_low=None, show_legend: bool = True):
    """Draw one frame: level bars in both panels and the arrows between them."""
    for ax in (ax_up, ax_low):
        ax.clear()
    xu = _bar_positions(data["m_up"], data["e_up"])
    xl = _bar_positions(data["m_low"], data["e_low"])
    for k, (x0, x1) in enumerate(xu):
        ax_up.hlines(data["e_up"][k], x0, x1, color="k", lw=2)
    for k, (x0, x1) in enumerate(xl):
        ax_low.hlines(data["e_low"][k], x0, x1, color="k", lw=2)
    for r in data["arrows"].itertuples():
        xa, xb = np.mean(xu[r.upper]), np.mean(xl[r.lower])
        arrow = ConnectionPatch(xyA=(xa, data["e_up"][r.upper]), coordsA=ax_up.transData,
                                xyB=(xb, data["e_low"][r.lower]), coordsB=ax_low.transData,
                                arrowstyle="-|>", mutation_scale=6, color=POL_COLORS[r.polarisation],
                                lw=max(0.3, LW_PER_INTENSITY * r.intensity), alpha=0.8)
        ax_low.figure.add_artist(arrow)
    pad = lambda e: (np.ptp(e) or 1.0) * 0.25 + 0.3
    ax_up.set_ylim(*(ylim_up or (data["e_up"].min() - pad(data["e_up"]), data["e_up"].max() + pad(data["e_up"]))))
    ax_low.set_ylim(*(ylim_low or (data["e_low"].min() - pad(data["e_low"]), data["e_low"].max() + pad(data["e_low"]))))
    for ax in (ax_up, ax_low):
        ax.set_xlim(-2.2, 2.2)
        ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
        ax.grid(alpha=0.2)
    ax_up.set_xticklabels([])
    ax_low.set_xlabel("M = ⟨J_z⟩ = m_l + m_s")
    ax_up.set_ylabel("upper manifold\n(cm⁻¹ from centre)")
    ax_low.set_ylabel("lower manifold\n(cm⁻¹ from centre)")
    if show_legend:
        for pol, col in POL_COLORS.items():
            ax_low.plot([], [], color=col, lw=3, label=POL_LABEL[pol])
        ax_low.legend(loc="upper left", fontsize=7)


def fig_grotrian(b_values=(0.1, 5.0, 60.0)):
    """Static diagram at a weak, an intermediate and a very strong field (Paschen-Back): arrows fade as B grows."""
    fig, axes = plt.subplots(2, len(b_values), figsize=(4.6 * len(b_values), 7), gridspec_kw={"hspace": 0.18})
    for j, b in enumerate(b_values):
        d = grotrian_data(b)
        draw_grotrian(axes[0, j], axes[1, j], d, show_legend=(j == 0))
        n_strong = int((d["arrows"].intensity > 0.2).sum())
        axes[0, j].set_title(f"B = {b:g} T\n{len(d['arrows'])} arrows, {n_strong} strong", fontsize=10)
    fig.suptitle("Sodium D lines as a level diagram: arrow colour = polarisation, width = intensity", fontsize=11)
    return fig


def animate_grotrian(save_path=None, b_max: float = 60.0, frames: int = 70):
    """GIF: B rises from near zero to b_max; levels fan out, arrow widths follow the intensities."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    bs = np.linspace(0.05, b_max, frames)
    fig, (ax_up, ax_low) = plt.subplots(2, 1, figsize=(7.5, 7.5), gridspec_kw={"hspace": 0.18})

    def draw(k):
        for art in list(fig.artists):
            art.remove()
        d = grotrian_data(bs[k])
        draw_grotrian(ax_up, ax_low, d)
        ax_up.set_title(f"Sodium D lines, B = {bs[k]:.1f} T   ({int((d['arrows'].intensity > 0.2).sum())} strong arrows; energy scale follows B)", fontsize=10)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=110, blit=False)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=9))
    return anim
