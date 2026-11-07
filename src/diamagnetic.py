r"""Diamagnetic term of hydrogen: H_dia = e^2 B^2 / (8 m_e) * (x^2 + y^2), first-order perturbation theory.

For a hydrogen state |n l m> (field along z):

    <x^2 + y^2> = <r^2> * <sin^2 theta>
    <r^2>        = n^2 (5 n^2 + 1 - 3 l (l + 1)) / 2     (units of a0^2)
    <sin^2 theta> = 2 (l^2 + l - 1 + m^2) / ((2 l - 1)(2 l + 3))   (l >= 1);  2/3 for l = 0

The shift grows as B^2 n^4, so it matters only for large n or strong fields. It is a first-order estimate:
it is not added to the Hamiltonians, it tells you when that would be needed.
"""
from __future__ import annotations

from scipy import constants as sc

A0 = sc.physical_constants["Bohr radius"][0]


def r2_hydrogenic(n: int, l: int) -> float:
    """<r^2> in units of a0^2."""
    return n ** 2 * (5 * n ** 2 + 1 - 3 * l * (l + 1)) / 2.0


def sin2_theta_mean(l: int, m: int) -> float:
    """<sin^2 theta> for the spherical harmonic Y_lm."""
    if l == 0:
        return 2.0 / 3.0
    return 2.0 * (l * l + l - 1 + m * m) / ((2 * l - 1) * (2 * l + 3))


def diamagnetic_shift_ev(n: int, l: int, m: int, b_tesla: float) -> float:
    """First-order diamagnetic shift (eV) of the hydrogen state |n l m>."""
    rho2 = r2_hydrogenic(n, l) * sin2_theta_mean(l, m) * A0 ** 2
    return sc.e ** 2 * b_tesla ** 2 / (8 * sc.m_e) * rho2 / sc.e


def balmer_alpha_diamagnetic_shift_cm1(b_tesla: float) -> dict:
    """Largest diamagnetic shift of the H-alpha line (n=3 -> 2) in cm^-1: upper state minus lower state, over all
    states (l, |m|) of the two levels. Returns the extreme values (min, max)."""
    from .constants import ev_to_wavenumber
    ups = [(3, l, m) for l in range(3) for m in range(-l, l + 1)]
    lows = [(2, l, m) for l in range(2) for m in range(-l, l + 1)]
    sh = [ev_to_wavenumber(diamagnetic_shift_ev(*u, b_tesla) - diamagnetic_shift_ev(*d, b_tesla))
          for u in ups for d in lows if abs(u[2] - d[2]) <= 1 and abs(u[1] - d[1]) == 1]
    return {"min": min(sh), "max": max(sh)}
