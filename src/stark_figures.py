"""Figures for hydrogen n = 2 in an electric field (src/stark.py)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .hydrogen import ev_to_mhz
from .stark import A_2P_1S, anticrossing_gap_mhz, h_n2_stark, lifetime_s, s_weights


def fig_stark_fan(e_max: float = 400.0, n_points: int = 200):
    """Left: the four lowest n = 2 levels against E (E parallel to B, B ~ 0), coloured by 2s content.
    Right: lifetime of the 2S-like state against E: mixing with 2P quenches it from 0.12 s to nanoseconds."""
    es = np.linspace(0.0, e_max, n_points)
    ref = np.linalg.eigvalsh(h_n2_stark(1e-6, 0.0, 0.0))[0]            # 2P1/2 at E = 0
    en, sw, lt = [], [], []
    for e in es:
        w, v = np.linalg.eigh(h_n2_stark(1e-6, e, 0.0))
        en.append(ev_to_mhz(w[:4] - ref))
        sw.append(s_weights(v)[:4])
        lt.append(lifetime_s((1 - s_weights(v)[2]) * A_2P_1S))        # level 2 is the S-like state at small E
    en, sw = np.array(en), np.array(sw)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for k in range(4):
        sc = ax1.scatter(es, en[:, k], c=sw[:, k], cmap="coolwarm", vmin=0, vmax=1, s=4)
    fig.colorbar(sc, ax=ax1, label="2s content of the state")
    ax1.set_xlabel("electric field E parallel to B (V/cm)")
    ax1.set_ylabel("energy above 2P1/2 at E = 0 (MHz)")
    ax1.set_title("2S1/2 and 2P1/2 repel and mix as E grows")
    ax2.semilogy(es[1:], lt[1:], "k")
    ax2.axhline(1.6e-9, color="tab:red", ls="--", lw=1)
    ax2.text(es[-1], 1.6e-9 * 1.6, "2P lifetime 1.6 ns", color="tab:red", ha="right", fontsize=8)
    ax2.axhline(0.1214, color="tab:blue", ls="--", lw=1)
    ax2.text(es[-1], 0.1214 * 1.6, "pure 2S about 0.12 s", color="tab:blue", ha="right", fontsize=8)
    ax2.set_xlabel("electric field E (V/cm)")
    ax2.set_ylabel("lifetime of the 2S-like state (s)")
    ax2.set_title("Quenching of the metastable 2S state")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_anticrossing(e_values=(0.0, 2.0, 5.0, 10.0), b_gauss=(525.0, 625.0), n: int = 240):
    """The 573 G crossing becomes an anticrossing when E is perpendicular to B; the gap grows in proportion to E."""
    bs = np.linspace(b_gauss[0], b_gauss[1], n) * 1e-4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(e_values)))
    # the pair that crosses: find the closest neighbouring levels at the crossing field
    w_c = ev_to_mhz(np.linalg.eigvalsh(h_n2_stark(573.4e-4, 0.0, 90.0)))
    k = int(np.argmin(np.diff(w_c)))
    for e, c in zip(e_values, cols):
        lv = np.array([ev_to_mhz(np.linalg.eigvalsh(h_n2_stark(b, e, 90.0)))[k:k + 2] for b in bs])
        ref = lv.mean()
        for j in (0, 1):
            ax1.plot(bs * 1e4, lv[:, j] - w_c[k:k + 2].mean(), color=c, label=f"E = {e:g} V/cm" if j == 0 else None)
    ax1.set_xlabel("magnetic field B (G)")
    ax1.set_ylabel("energy relative to the crossing (MHz)")
    ax1.set_title("2S1/2(-1/2) and 2P1/2(+1/2): crossing → anticrossing")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    es = np.linspace(0.5, 20, 12)
    gaps = [anticrossing_gap_mhz(e)[0] for e in es]
    ax2.plot(es, gaps, "o-")
    ax2.set_xlabel("electric field E perpendicular to B (V/cm)")
    ax2.set_ylabel("minimum gap (MHz)")
    ax2.set_title("Gap is proportional to E")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig
