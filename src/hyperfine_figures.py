"""Figures for hyperfine structure (src/hyperfine.py)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .hydrogen import ev_to_mhz, h_n2
from .hyperfine import h_n2_hyperfine, h_na_3s, lamb_crossings_gauss
from .solvers import track_from_builder


def fig_hyperfine(n: int = 301):
    """Left: sodium 3S1/2 hyperfine levels (Breit-Rabi diagram, 8 levels). Right: hydrogen 2S/2P levels near
    573 G; without hyperfine one crossing (dashed), with hyperfine four. 2P3/2 lies ~11 GHz higher (cropped)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    bs = np.linspace(0.0, 0.3, 200)
    e = track_from_builder(h_na_3s, bs)
    for k in range(e.shape[1]):
        ax1.plot(bs * 1e4, ev_to_mhz(e[:, k]), lw=1.2)
    ax1.set_xlabel("B (gauss)"); ax1.set_ylabel("energy (MHz)")
    ax1.set_title("Na 3S1/2 (I = 3/2): F = 1 and F = 2, then m_J = ±1/2 groups")
    bw = np.linspace(0.050, 0.065, 301)
    e0 = track_from_builder(h_n2, bw)
    e1 = track_from_builder(h_n2_hyperfine, bw)
    for k in range(e0.shape[1]):
        ax2.plot(bw * 1e4, ev_to_mhz(e0[:, k]), "k--", lw=0.8)
    for k in range(e1.shape[1]):
        ax2.plot(bw * 1e4, ev_to_mhz(e1[:, k]), lw=0.9)
    xs = lamb_crossings_gauss(hyperfine=True)
    for x in xs:
        ax2.axvline(x, color="gray", lw=0.5)
    ax2.set_ylim(-450, 2150)
    ax2.set_xlabel("B (gauss)"); ax2.set_ylabel("energy (MHz)")
    ax2.set_title("H n = 2: one crossing (dashed) becomes four; mean %.1f G" % np.mean(xs))
    fig.tight_layout()
    return fig
