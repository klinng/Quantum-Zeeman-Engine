r"""Polarisation optics with Jones calculus, and the polarisation of Zeeman components.

Conventions
-----------
* Field along z. A line of sight (towards the observer) makes angle theta with B.
* Jones vectors use the basis (e_theta, e_phi): e_phi = y is perpendicular to B and to the line of
  sight, e_theta lies in the plane of B and the line of sight (it is the projection of B on the sky).
* Time dependence exp(-i omega t). With this convention a Zeeman component with
  q = m_upper - m_lower is emitted by a radiating dipole d_q:

      pi     (q = 0):   d = z                          Jones (-sin t, 0)
      sigma+ (q = +1):  d = (x + i y)/sqrt(2)          Jones ( cos t, +i)/sqrt(2)
      sigma- (q = -1):  d = (x - i y)/sqrt(2)          Jones ( cos t, -i)/sqrt(2)

  (sigma+ carries +hbar of angular momentum along B; seen by an observer looking back at the source
  along B it rotates counter-clockwise.)
* The squared length of each Jones vector is the angular intensity pattern:
  |J_pi|^2 = sin^2 t and |J_sigma|^2 = (1 + cos^2 t)/2. Both integrate to 8 pi / 3 over the sphere,
  so a component's total emission rate is shared out by direction without gain or loss.
"""
from __future__ import annotations

import numpy as np


def component_jones(q: int, theta_rad: float) -> np.ndarray:
    """Jones vector (in the (e_theta, e_phi) basis) of a Zeeman component seen at angle theta from B."""
    if q == 0:
        return np.array([-np.sin(theta_rad), 0.0], dtype=complex)
    return np.array([np.cos(theta_rad), 1j * q], dtype=complex) / np.sqrt(2)


def angular_intensity(q: int, theta_rad: float) -> float:
    """Relative intensity of a component per unit solid angle: sin^2 t (pi) or (1 + cos^2 t)/2 (sigma)."""
    return float(np.vdot(component_jones(q, theta_rad), component_jones(q, theta_rad)).real)


def linear_polariser(phi_deg: float) -> np.ndarray:
    """Ideal linear polariser with transmission axis at phi from e_theta (phi = 0: parallel to B on the sky)."""
    p = np.radians(phi_deg)
    c, s = np.cos(p), np.sin(p)
    return np.array([[c * c, c * s], [c * s, s * s]], dtype=complex)


def circular_polariser(handedness: int) -> np.ndarray:
    """Ideal analyser that passes only sigma+ (handedness = +1) or sigma- (-1) light viewed along B."""
    c = np.array([1.0, 1j * handedness]) / np.sqrt(2)
    return np.outer(c, c.conj())


def quarter_wave_plate(fast_axis_deg: float) -> np.ndarray:
    """Quarter-wave retarder with its fast axis at the given angle (slow axis delayed by pi/2)."""
    p = np.radians(fast_axis_deg)
    rot = np.array([[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]])
    return rot @ np.diag([1, 1j]) @ rot.T


def transmitted_intensity(jones: np.ndarray, element: np.ndarray) -> float:
    """Intensity |M J|^2 after an optical element (or product of elements) M."""
    out = element @ jones
    return float(np.vdot(out, out).real)


def polarisation_ellipse(jones: np.ndarray, n: int = 200) -> np.ndarray:
    """Points (E_theta, E_phi) traced by the real field Re[J exp(-i omega t)] over one period."""
    t = np.linspace(0, 2 * np.pi, n)
    field = np.real(jones[:, None] * np.exp(-1j * t)[None, :])
    return field.T
