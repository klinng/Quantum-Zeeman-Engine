"""How fine must a spectrometer be to see the Zeeman components?

Resolving power R = lambda / d(lambda). The closest sodium Zeeman components are only ~(2/3) mu_B B apart,
so the field you need is set by the instrument. These are IDEAL (theoretical) powers; real instruments
reach less. Each function states the formula it uses.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..constants import MU_B_EV
from .prism import prism_resolving_power
from .spectrometer import field_for_resolving_power, required_resolving_power, sodium_d_components

B_STAR_TESLA = 24.6        # weak-to-strong field crossover of the sodium 3p level (zeta / mu_B)


def grating_resolving_power(lines_per_mm: float, width_mm: float, order: int = 1) -> float:
    """R = m N, with N the number of illuminated grating lines."""
    return float(order * lines_per_mm * width_mm)


def fabry_perot_resolving_power(gap_mm: float, finesse: float, wavelength_nm: float = 589.3) -> float:
    """R ~ m F, with interference order m = 2 d / lambda (air gap, normal incidence)."""
    return float(2.0 * gap_mm * 1e6 / wavelength_nm * finesse)


def instrument_ladder(wavelength_nm: float = 589.3) -> dict[str, float]:
    """Ideal resolving powers of typical instruments, in increasing order."""
    return {
        "Prism (BK7, 5 cm base)": prism_resolving_power(wavelength_nm, 50, "BK7"),
        "Prism (SF11, 5 cm base)": prism_resolving_power(wavelength_nm, 50, "SF11"),
        "Grating (600 l/mm, 5 cm)": grating_resolving_power(600, 50),
        "Large grating (1200 l/mm, 10 cm)": grating_resolving_power(1200, 100),
        "Fabry-Perot (10 mm gap, finesse 30)": fabry_perot_resolving_power(10.0, 30.0, wavelength_nm),
    }


def required_power_vs_field(b_values) -> np.ndarray:
    """Resolving power needed to separate the closest sodium Zeeman components at each field (T)."""
    return np.array([required_resolving_power(sodium_d_components(float(b))) for b in b_values])


def fig_instrument_ladder(b_values=(0.05, 0.1, 0.5, 1.0, 2.0, 5.0)):
    """Left: resolving power of each instrument against what the Zeeman pattern needs at several fields.
    Right: the weakest field at which each instrument can just separate the closest components."""
    ladder = instrument_ladder()
    b = np.array(b_values, dtype=float)
    need = required_power_vs_field(b)
    names, powers = list(ladder), np.array(list(ladder.values()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    y = np.arange(len(names))
    ax1.barh(y, powers, color="0.7", edgecolor="k")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xscale("log")
    for bi, ni in zip(b, need):
        ax1.axvline(ni, color="tab:red", ls="--", lw=1)
        ax1.annotate(f"{bi:g} T", xy=(ni, 0.02), xycoords=("data", "axes fraction"), rotation=90,
                     color="tab:red", fontsize=8, ha="right", va="bottom")
    ax1.set_xlabel("resolving power  R = λ/Δλ")
    ax1.set_title("Which instrument can separate the lines?\n(red dashes: R needed at that field)", fontsize=10)
    ax1.grid(alpha=0.3, axis="x")

    bmin = np.array([field_for_resolving_power(r) for r in powers])
    ax2.barh(y, bmin, color=["tab:red" if v > B_STAR_TESLA * 0.8 else "tab:blue" for v in bmin], edgecolor="k")
    ax2.set_xscale("log")
    ax2.axvline(B_STAR_TESLA, color="k", ls=":", lw=1.2)
    ax2.text(B_STAR_TESLA * 1.15, 2.0, "B* = 24.6 T\nweak-field\npicture ends", fontsize=8, ha="left", va="center")
    for yi, v in zip(y, bmin):
        ax2.text(v * 1.12, yi, f"{v:.3g} T", va="center", fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(left=0.02, right=300)
    ax2.set_xlabel("weakest resolvable field (T)")
    ax2.set_title("What field does each instrument need?\n(closest Zeeman components just separated)", fontsize=10)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def zeeman_shift_nm(b_tesla: float, wavelength_nm: float = 589.3, g_times_m: float = 5 / 3) -> float:
    """Wavelength shift (nm) of a Zeeman component with g*m_j = g_times_m (5/3 = outermost D2 line)."""
    return g_times_m * MU_B_EV / 4.135667696e-15 / 2.99792458e10 * wavelength_nm ** 2 * 1e-7 * b_tesla
