"""Can an instrument separate the hydrogen H-alpha Zeeman groups?

Criterion used here: the sigma groups sit +-20.11 pm/T x B from the pi group (normal Zeeman limit). An instrument of
resolving power R separates two lines that are lambda/R apart, so it needs R >= lambda / (20.11 pm/T x B).
Two honest extras:
* Doppler floor. At 300 K each hydrogen line is already 8.1 pm wide, so the lines cannot be sharper than that
  whatever the instrument: resolving power beyond lambda / FWHM_Doppler (about 81 000) buys nothing.
* Hydrogen is light, so its Doppler width is large; a heavy atom or a cold source would not have this limit.
All powers are ideal (theoretical); real instruments reach less.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..hydrogen_spectrum import doppler_fwhm_pm
from .instruments import fabry_perot_resolving_power, grating_resolving_power
from .prism import deviation_deg, min_deviation_incidence_deg, prism_resolving_power

LAMBDA_HALPHA_NM = 656.28
NORMAL_ZEEMAN_PM_PER_T = 20.11


def halpha_instrument_ladder() -> dict[str, float]:
    """Ideal resolving powers of typical instruments at H-alpha."""
    w = LAMBDA_HALPHA_NM
    return {
        "Prism (BK7, 5 cm base)": prism_resolving_power(w, 50, "BK7"),
        "Prism (SF11, 5 cm base)": prism_resolving_power(w, 50, "SF11"),
        "Grating (600 l/mm, 5 cm)": grating_resolving_power(600, 50),
        "Large grating (1200 l/mm, 10 cm)": grating_resolving_power(1200, 100),
        "Fabry-Perot (10 mm gap, finesse 30)": fabry_perot_resolving_power(10.0, 30.0, w),
    }


def required_power_halpha(b_tesla: float) -> float:
    """Resolving power needed to separate the sigma groups from the pi group at field B."""
    return LAMBDA_HALPHA_NM / (NORMAL_ZEEMAN_PM_PER_T * 1e-3 * b_tesla)


def weakest_field_halpha(resolving_power: float) -> float:
    """Weakest field (T) at which an ideal instrument of this power just separates the groups."""
    return LAMBDA_HALPHA_NM / (NORMAL_ZEEMAN_PM_PER_T * 1e-3 * resolving_power)


def doppler_limited_power(temperature_k: float = 300.0) -> float:
    """Resolving power beyond which a hydrogen lamp's thermal width, not the instrument, sets the sharpness."""
    return LAMBDA_HALPHA_NM / (doppler_fwhm_pm(temperature_k=temperature_k) * 1e-3)


def prism_zeeman_deflection_urad(b_tesla: float, glass: str = "SF11", apex_deg: float = 60.0) -> float:
    """Angle (microradians) between the two sigma groups' rays after a prism at minimum deviation, at field B."""
    inc = min_deviation_incidence_deg(LAMBDA_HALPHA_NM, apex_deg, glass)
    half = NORMAL_ZEEMAN_PM_PER_T * 1e-3 * b_tesla
    d_hi = deviation_deg(LAMBDA_HALPHA_NM + half, inc, apex_deg, glass)
    d_lo = deviation_deg(LAMBDA_HALPHA_NM - half, inc, apex_deg, glass)
    return float(abs(d_lo - d_hi) * np.pi / 180 * 1e6)


def fig_halpha_instrument_ladder(b_values=(0.5, 1.0, 2.0, 5.0)):
    """Left: instrument powers against what H-alpha needs (red dashes) and the 300 K Doppler floor (black).
    Right: the weakest field each instrument needs, never lower than the Doppler floor."""
    ladder = halpha_instrument_ladder()
    names, powers = list(ladder), np.array(list(ladder.values()))
    dop = doppler_limited_power()
    y = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax1.barh(y, powers, color="0.7", edgecolor="k")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xscale("log")
    for b in b_values:
        r = required_power_halpha(b)
        ax1.axvline(r, color="tab:red", ls="--", lw=1)
        ax1.annotate(f"{b:g} T", xy=(r, 0.02), xycoords=("data", "axes fraction"), rotation=90,
                     color="tab:red", fontsize=8, ha="right", va="bottom")
    ax1.axvline(dop, color="k", lw=1.5)
    ax1.text(dop * 0.93, 0.5, "300 K Doppler floor:\nnothing sharper is visible", fontsize=8, va="center", ha="right", bbox=dict(fc="white", ec="none", alpha=0.8))
    ax1.set_xlabel("resolving power  R = λ/Δλ")
    ax1.set_title("H-alpha: which instrument separates the σ and π groups?\n(red dashes: R needed at that field)", fontsize=10)
    ax1.grid(alpha=0.3, axis="x")

    b_dop = weakest_field_halpha(dop)
    bmin = np.array([max(weakest_field_halpha(r), b_dop) for r in powers])     # the Doppler floor cannot be beaten
    limited = np.array([weakest_field_halpha(r) < b_dop for r in powers])
    ax2.barh(y, bmin, color=["tab:orange" if l else "tab:blue" for l in limited], edgecolor="k")
    ax2.set_xscale("log")
    ax2.axvline(b_dop, color="k", lw=1.5)
    ax2.text(b_dop * 0.9, len(names) - 0.45, f"Doppler floor {b_dop:.2g} T", fontsize=8, va="center", ha="right")
    for yi, v in zip(y, bmin):
        ax2.text(v * 1.12, yi, f"{v:.3g} T" + ("  (Doppler-limited)" if limited[yi] else ""), va="center", fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(left=0.02, right=200)
    ax2.set_xlabel("weakest field for a clean separation (T)")
    ax2.set_title("What field does each instrument need?\n(σ groups just separated, 300 K hydrogen)", fontsize=10)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig
