"""Error bands from input uncertainties, drawn on real model curves.

The result is honest rather than flattering: with the published input uncertainties, the bands are far narrower
than the spectral lines themselves, so the dominant limit is the model (what it leaves out), not the input numbers.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .hydrogen import ev_to_mhz, h_n2
from .hyperfine import NA_A_MHZ, NA_G_I, NA_I, h_with_hyperfine, mhz_to_ev
from .uncertainty import curve_band


def _h_s_energy(b_t, lamb_mhz):
    """2S1/2 (m_j = -1/2) energy (MHz) against field."""
    return np.array([ev_to_mhz(np.linalg.eigvalsh(h_n2(x, lamb_mhz=lamb_mhz)[:2, :2])[0]) for x in b_t])


def _na_3p12_levels(b_t, a_3p12):
    """Lowest and highest 3P1/2 hyperfine-Zeeman levels (MHz): returns shape (len(b), 2)."""
    from .validation import zeta_from_fine_structure
    zeta = zeta_from_fine_structure(17.196, l=1)
    a = {0.5: mhz_to_ev(a_3p12), 1.5: mhz_to_ev(NA_A_MHZ["3P3/2"])}
    out = []
    for x in b_t:
        w = np.sort(np.linalg.eigvalsh(h_with_hyperfine(1, NA_I, a, x, NA_G_I, zeta, e0_ev=zeta)))[:8]
        out.append([ev_to_mhz(w[0]), ev_to_mhz(w[-1])])
    return np.array(out)


def fig_error_bands(n: int = 150):
    """Left: hydrogen 2S level near the 573 G crossing with its band from the measured Lamb-shift uncertainty
    (the band is shown magnified 1000 times, otherwise it would be invisible). Right: the 1-sigma width of two sodium
    3P1/2 hyperfine levels from A(3P1/2) = 94.44(13) MHz, against field."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    b = np.linspace(0.050, 0.065, 31)
    band = curve_band(_h_s_energy, {"lamb_mhz": (1057.845, 0.009)}, b, n=40)
    e_p = np.array([ev_to_mhz(np.linalg.eigvalsh(h_n2(x)[2:, 2:])[1]) for x in b])
    a1.plot(b * 1e4, e_p, color="tab:green", label="2P1/2 (m_j = +1/2)")
    a1.plot(b * 1e4, band["mean"], color="tab:blue", label="2S1/2 (m_j = −1/2)")
    a1.fill_between(b * 1e4, band["mean"] - 1000 * band["std"], band["mean"] + 1000 * band["std"], color="tab:blue", alpha=0.3,
                    label=f"±1σ band ×1000 (true σ = {band['std'].mean() * 1e3:.1f} kHz)")
    a1.set_xlabel("B (gauss)"); a1.set_ylabel("energy (MHz)"); a1.legend(fontsize=8)
    a1.set_title("H crossing: Lamb-shift uncertainty 0.009 MHz moves the 2S level\nby far less than the 45 to 110 MHz line widths of Westig et al.", fontsize=9)
    bb = np.linspace(0.0, 0.1, 21)
    band2 = curve_band(_na_3p12_levels, {"a_3p12": (94.44, 0.13)}, bb, n=n)
    for k, lab in enumerate(("lowest level", "highest level")):
        a2.plot(bb * 1e4, band2["std"][:, k], lw=2, label=lab)
    a2.axhline(9.8, color="gray", ls="--", label="Na D2 natural width 9.8 MHz")
    a2.set_xlabel("B (gauss)"); a2.set_ylabel("1σ uncertainty of the level (MHz)"); a2.legend(fontsize=8)
    a2.set_title("Sodium 3P1/2 hyperfine levels: uncertainty from A(3P1/2)\nis tenths of a MHz, below the natural width", fontsize=9)
    fig.tight_layout()
    return fig
