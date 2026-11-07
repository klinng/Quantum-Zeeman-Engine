r"""Isotope shift from the reduced mass: the electron orbits the common centre of mass, so the effective Rydberg constant is

    R_X = R_inf / (1 + m_e / M_X)       (M_X = nuclear mass of isotope X)

Hydrogen (proton) and deuterium (deuteron) therefore differ in every line. Only this mass effect is included;
the finite nuclear size is not, and is much smaller for hydrogen isotopes. Masses and R_inf are CODATA values
from scipy.constants.
"""
from __future__ import annotations

from scipy import constants as sc

R_INF_CM1 = sc.physical_constants["Rydberg constant"][0] / 100.0
M_E = sc.m_e
NUCLEAR_MASS = {"H": sc.m_p, "D": sc.physical_constants["deuteron mass"][0]}


def rydberg_cm1(isotope: str) -> float:
    """Rydberg constant (cm^-1) for hydrogen ('H') or deuterium ('D')."""
    return R_INF_CM1 / (1 + M_E / NUCLEAR_MASS[isotope])


def line_wavenumber_cm1(isotope: str, n_up: int = 3, n_low: int = 2) -> float:
    """Gross-structure vacuum wavenumber of the n_up -> n_low line (no fine structure, no Lamb shift)."""
    return rydberg_cm1(isotope) * (1 / n_low ** 2 - 1 / n_up ** 2)


def isotope_shift(n_up: int = 3, n_low: int = 2) -> dict:
    """Deuterium minus hydrogen: wavenumber (cm^-1), vacuum wavelength (pm, negative: D is bluer)."""
    kh, kd = line_wavenumber_cm1("H", n_up, n_low), line_wavenumber_cm1("D", n_up, n_low)
    lam_h, lam_d = 1e7 / kh, 1e7 / kd                     # nm
    return {"wavenumber_cm1": kd - kh, "wavelength_pm": (lam_d - lam_h) * 1e3,
            "lambda_h_nm": lam_h, "lambda_d_nm": lam_d}


def zeeman_shift_pm_per_t(isotope: str = "H", n_up: int = 3, n_low: int = 2) -> float:
    """Normal-Zeeman shift scale mu_B B (1 T), in pm: it is nearly the same for H and D (it depends on lambda^2)."""
    lam_m = 1e-2 / line_wavenumber_cm1(isotope, n_up, n_low)
    return lam_m ** 2 * sc.physical_constants["Bohr magneton in inverse meters per tesla"][0] * 1e12
