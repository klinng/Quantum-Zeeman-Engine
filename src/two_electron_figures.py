"""Figures for the two-electron model (src/two_electron.py)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .constants import ev_to_wavenumber
from .two_electron import classify_states

COLORS = {"3P0": "tab:gray", "3P1": "tab:blue", "3P2": "tab:green", "1P1": "tab:red"}


def fig_two_electron_levels(zeta_ev: float, delta_ev: float, e_t_ev: float, b_max: float = 20.0, n: int = 81):
    """Left: all 12 levels of the s-p configuration against B (energy above the lowest level), coloured by term.
    Right: Zeeman splitting of the 3P1, 3P2 and 1P1 levels (zero-field centre removed), showing the slopes g_J mu_B B."""
    bs = np.linspace(0, b_max, n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    e_low = classify_states(0.0, zeta_ev, delta_ev, e_t_ev).energy_ev.min()
    for lab in COLORS:
        rows = []
        for b in bs:
            df = classify_states(b, zeta_ev, delta_ev, e_t_ev)
            g = df[df.label == lab].sort_values("energy_ev")
            rows.append(ev_to_wavenumber(g.energy_ev.values - e_low))
        rows = np.array(rows)
        if lab != "1P1":                       # the singlet lies thousands of cm^-1 higher: it would flatten the triplet fans
            ax1.plot(bs, rows, color=COLORS[lab], lw=1.3)
        if lab != "3P0":
            ax2.plot(bs, rows - rows[0].mean(), color=COLORS[lab], lw=1.3)
        if lab != "1P1":
            ax1.plot([], [], color=COLORS[lab], label=lab)
    ax1.set_xlabel("magnetic field B (T)")
    ax1.set_ylabel("energy above the lowest level (cm$^{-1}$)")
    ax1.set_title("The three 3P levels (1P1 lies about 8 400 cm$^{-1}$ higher, off scale)", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("magnetic field B (T)")
    ax2.set_ylabel("energy relative to the zero-field centre (cm$^{-1}$)")
    ax2.set_title("Zeeman fans: 3P1 and 3P2 share g ≈ 1.5, 1P1 has g = 1")
    fig.tight_layout()
    return fig
