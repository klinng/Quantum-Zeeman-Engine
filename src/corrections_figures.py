"""Figure for the small corrections: diamagnetic term and the H/D isotope shift (src/diamagnetic.py, src/isotope.py)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .diamagnetic import balmer_alpha_diamagnetic_shift_cm1
from .hydrogen_spectrum import doppler_fwhm_pm
from .isotope import isotope_shift, zeeman_shift_pm_per_t

LAMBDA_NM = 656.47


def _cm1_to_pm(cm1: float) -> float:
    return cm1 * 1e-7 * LAMBDA_NM ** 2 * 1e3          # d(lambda) = lambda^2 d(nu~), nm -> pm


def fig_corrections():
    """Left: largest diamagnetic shift of H-alpha against field, with the Zeeman shift and Doppler width for scale.
    Right: the H/D isotope shift against the Zeeman shift of one component and the Doppler width."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    b = np.logspace(0, np.log10(80), 40)
    dia = np.array([_cm1_to_pm(balmer_alpha_diamagnetic_shift_cm1(x)["max"]) for x in b])
    a1.loglog(b, dia, lw=2, label="diamagnetic (largest component)")
    a1.loglog(b, zeeman_shift_pm_per_t() * b, label="Zeeman scale μB·B")
    a1.axhline(doppler_fwhm_pm(temperature_k=300.0), color="gray", ls="--", label="Doppler FWHM, 300 K")
    a1.set_xlabel("B (T)"); a1.set_ylabel("shift (pm)"); a1.legend(fontsize=8)
    a1.set_title("H-alpha: diamagnetic term is small against Zeeman,\nbut reaches the Doppler width near 60 T", fontsize=10)
    s = isotope_shift()
    names = ["H/D isotope shift", "Zeeman μB·B at 1 T", "Zeeman μB·B at 10 T", "Doppler FWHM (H, 300 K)"]
    vals = [-s["wavelength_pm"], zeeman_shift_pm_per_t() * 1, zeeman_shift_pm_per_t() * 10, doppler_fwhm_pm(temperature_k=300.0)]
    a2.barh(names, vals, color=["tab:purple", "tab:blue", "tab:blue", "gray"])
    a2.set_xlabel("pm"); a2.set_title(f"Deuterium H-alpha is {-s['wavelength_pm']:.0f} pm bluer than hydrogen", fontsize=10)
    fig.tight_layout()
    return fig
