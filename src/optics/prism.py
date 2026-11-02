"""Glass prism: dispersion (Sellmeier), deviation angle, screen position and resolving power."""
from __future__ import annotations

import numpy as np

# Sellmeier coefficients (B1, B2, B3), (C1, C2, C3) with wavelength in micrometres (Schott data).
SELLMEIER = {
    "BK7": ((1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)),
    "SF11": ((1.73759695, 0.313747346, 1.89878101), (0.013188707, 0.0623068142, 155.23629)),
}


def refractive_index(wavelength_nm, glass: str = "BK7"):
    """Refractive index n(lambda) from the Sellmeier equation."""
    l2 = (np.asarray(wavelength_nm, dtype=float) / 1000.0) ** 2
    b, c = SELLMEIER[glass]
    return np.sqrt(1 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def deviation_deg(wavelength_nm, incidence_deg: float, apex_deg: float = 60.0, glass: str = "BK7"):
    """Total deviation angle (degrees) of a ray through the prism (Snell's law at both faces).

    Returns NaN where the ray is totally internally reflected at the second face.
    """
    n = refractive_index(wavelength_nm, glass)
    apex = np.radians(apex_deg)
    t1 = np.radians(incidence_deg)
    t1p = np.arcsin(np.sin(t1) / n)
    t2p = apex - t1p
    s = n * np.sin(t2p)
    t2 = np.arcsin(np.where(np.abs(s) <= 1, s, np.nan))
    return np.degrees(t1 + t2 - apex)


def min_deviation_incidence_deg(wavelength_nm: float, apex_deg: float = 60.0, glass: str = "BK7") -> float:
    """Incidence angle for which the ray passes symmetrically (minimum deviation) at this wavelength."""
    n = float(refractive_index(wavelength_nm, glass))
    return float(np.degrees(np.arcsin(n * np.sin(np.radians(apex_deg) / 2))))


def min_deviation_deg(wavelength_nm: float, apex_deg: float = 60.0, glass: str = "BK7") -> float:
    """Minimum deviation angle: delta_min = 2 asin(n sin(A/2)) - A."""
    return 2 * min_deviation_incidence_deg(wavelength_nm, apex_deg, glass) - apex_deg


def screen_position_m(wavelength_nm, distance_m: float = 1.0, ref_nm: float = 589.3,
                      apex_deg: float = 60.0, glass: str = "BK7"):
    """Position on a screen (metres from the reference colour) with the prism set at minimum
    deviation for ref_nm. Shorter wavelengths bend more, so they land at larger positive values."""
    inc = min_deviation_incidence_deg(ref_nm, apex_deg, glass)
    delta = deviation_deg(wavelength_nm, inc, apex_deg, glass)
    delta_ref = min_deviation_deg(ref_nm, apex_deg, glass)
    return distance_m * np.tan(np.radians(delta - delta_ref))


def prism_resolving_power(wavelength_nm: float, base_mm: float = 50.0, glass: str = "BK7") -> float:
    """Resolving power R = lambda / d(lambda) = b |dn/dlambda| for a prism with base length b."""
    h = 0.5
    dn = float(refractive_index(wavelength_nm + h, glass) - refractive_index(wavelength_nm - h, glass))
    return base_mm * 1e6 * abs(dn / (2 * h))          # base converted to nm
