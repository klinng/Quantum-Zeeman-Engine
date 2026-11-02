"""Wavelength to display colour. An approximation for plots, not a colorimetric standard."""
from __future__ import annotations

import numpy as np


def wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> tuple[float, float, float]:
    """Approximate RGB (0..1) of a visible wavelength; black outside 380-750 nm.

    Piecewise-linear approximation (after D. Bruton) with dimming toward the violet and red ends.
    """
    w = float(wavelength_nm)
    if 380 <= w < 440:
        r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / 50, 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif 510 <= w < 580:
        r, g, b = (w - 510) / 70, 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / 65, 0.0
    elif 645 <= w <= 750:
        r, g, b = 1.0, 0.0, 0.0
    else:
        return (0.0, 0.0, 0.0)
    if w < 420:
        f = 0.3 + 0.7 * (w - 380) / 40
    elif w <= 700:
        f = 1.0
    else:
        f = 0.3 + 0.7 * (750 - w) / 50
    return tuple(float((c * f) ** gamma) if c > 0 else 0.0 for c in (r, g, b))


def rainbow_strip(lo_nm: float = 380, hi_nm: float = 750, width: int = 600, height: int = 20) -> np.ndarray:
    """RGB image (height, width, 3) of the visible spectrum, for use as a plot background."""
    wl = np.linspace(lo_nm, hi_nm, width)
    row = np.array([wavelength_to_rgb(x) for x in wl])
    return np.repeat(row[None, :, :], height, axis=0)
