r"""Line shapes: Doppler (Gaussian) + natural and pressure (Lorentzian) = Voigt.

* Doppler width from the thermal motion of the atoms: Gaussian, FWHM ~ sqrt(T / mass).
* Natural width from the finite lifetime: Lorentzian, FWHM = (A_upper + A_lower) / (2 pi), with A the total decay rates.
* Pressure (collision) width: Lorentzian, proportional to the pressure. No coefficient is built in: you pass the width
  you want to explore, because no checked collision coefficients are available.
* Voigt profile = their convolution, ``scipy.special.voigt_profile``. Every profile has unit area.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

from .hyperfine_lines import broadened_spectrum, doppler_fwhm_mhz, sodium_d_hyperfine_lines

LN2 = np.log(2.0)
# Na 3P decay rates from the NIST Handbook (persistent lines table, ref. M03), in 1/s
NA_A_D2 = 0.616e8
NA_A_D1 = 0.614e8


def natural_fwhm_mhz(a_upper_per_s: float, a_lower_per_s: float = 0.0) -> float:
    """Natural (lifetime) FWHM in MHz: (A_up + A_low) / (2 pi)."""
    return (a_upper_per_s + a_lower_per_s) / (2 * np.pi) * 1e-6


def voigt_mhz(freq_mhz, centre_mhz: float, fwhm_gauss_mhz: float, fwhm_lorentz_mhz: float):
    """Area-normalised Voigt profile (1/MHz) from a Gaussian and a Lorentzian FWHM."""
    sigma = fwhm_gauss_mhz / (2 * np.sqrt(2 * LN2))
    gamma = fwhm_lorentz_mhz / 2.0                       # Lorentzian half width at half maximum
    x = np.asarray(freq_mhz, dtype=float) - centre_mhz
    if sigma == 0.0:
        return gamma / np.pi / (x ** 2 + gamma ** 2)
    return voigt_profile(x, sigma, gamma)


def voigt_fwhm_approx(fwhm_gauss: float, fwhm_lorentz: float) -> float:
    """Olivero-Longbothum approximation: 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)."""
    return 0.5346 * fwhm_lorentz + np.sqrt(0.2166 * fwhm_lorentz ** 2 + fwhm_gauss ** 2)


def voigt_fwhm_exact(fwhm_gauss: float, fwhm_lorentz: float) -> float:
    """Numerical FWHM of the Voigt profile (root of profile(x) = peak / 2)."""
    peak = voigt_mhz(0.0, 0.0, fwhm_gauss, fwhm_lorentz)
    hi = voigt_fwhm_approx(fwhm_gauss, fwhm_lorentz)
    return 2 * brentq(lambda x: voigt_mhz(x, 0.0, fwhm_gauss, fwhm_lorentz) - peak / 2, 0.0, hi)


def voigt_spectrum(df, line: str, fwhm_gauss_mhz: float, fwhm_lorentz_mhz: float, freq):
    """Sum of Voigt profiles over the components of one line (area of each = its strength)."""
    d = df[df.line == line]
    return sum(s * voigt_mhz(freq, f0, fwhm_gauss_mhz, fwhm_lorentz_mhz) for f0, s in zip(d.freq_mhz, d.strength))


def fig_voigt(pressure_fwhm_mhz: float = 50.0):
    """Left: sodium D2 hyperfine lines at 300 K, 1 K and 1 mK. Right: Voigt width from Gaussian and Lorentzian parts."""
    gam = natural_fwhm_mhz(NA_A_D2)
    d0 = sodium_d_hyperfine_lines(0.0)
    d2 = d0[d0.line == "D2"]
    centre = float(d2.freq_mhz @ d2.strength / d2.strength.sum())
    d0 = d0.assign(freq_mhz=d0.freq_mhz - centre)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    f = np.linspace(-1500, 1800, 4000)
    for temp, c in ((300.0, "tab:orange"), (1.0, "tab:green"), (1e-3, "tab:blue")):
        g = doppler_fwhm_mhz(temp)
        prof = voigt_spectrum(d0, "D2", g, gam, f)
        a1.plot(f, prof / prof.max(), color=c, label=f"T = {temp:g} K: Doppler {g:.3g} MHz")
    a1.set_xlabel("frequency from D2 centre of gravity (MHz)"); a1.set_ylabel("normalised intensity")
    a1.set_title(f"Na D2 hyperfine lines; natural width {gam:.1f} MHz", fontsize=10); a1.legend(fontsize=8)
    lz = np.logspace(-2, 3, 60)
    for fg, c in ((1.0, "tab:blue"), (30.0, "tab:green"), (1300.0, "tab:orange")):
        a2.loglog(lz, [voigt_fwhm_exact(fg, x) for x in lz], color=c, label=f"Gaussian {fg:g} MHz")
        a2.loglog(lz, [voigt_fwhm_approx(fg, x) for x in lz], "k:", lw=0.8)
    a2.axvline(gam, color="gray", ls="--"); a2.text(gam, 2, "Na D2 natural", rotation=90, fontsize=8, va="bottom")
    a2.axvline(pressure_fwhm_mhz, color="gray", ls=":"); a2.text(pressure_fwhm_mhz, 2, "50 MHz (example pressure width)", rotation=90, fontsize=8, va="bottom")
    a2.set_xlabel("Lorentzian FWHM (natural + pressure) (MHz)"); a2.set_ylabel("Voigt FWHM (MHz)")
    a2.set_title("Voigt width: exact (colour) and approximation (dotted)", fontsize=10); a2.legend(fontsize=8)
    fig.tight_layout()
    return fig
