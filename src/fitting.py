"""Fit physical parameters (spin-orbit constant zeta, optionally g_s) to measured energy levels.

This uses scipy.optimize.least_squares, so it needs no PyTorch. The optional torch_engine.py does
the same job with autograd; both must agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .constants import G_S
from .solvers import energy_vs_field


@dataclass
class FitResult:
    zeta_ev: float
    g_s: float
    rms_residual_ev: float
    success: bool


def fit_parameters(l: int, b_values, measured_levels, zeta0_ev: float, g_s0: float = G_S,
                   fit_g_s: bool = False) -> FitResult:
    """Least-squares fit of zeta (and optionally g_s).

    measured_levels : array (n_B, 2*(2l+1)) of energies in eV, each row sorted ascending.
    The parameters are rescaled internally (zeta/zeta0, g_s/g_s0) so the optimiser sees O(1) numbers.
    """
    b_values = np.asarray(b_values, dtype=float)
    measured = np.asarray(measured_levels, dtype=float)

    def unpack(p):
        zeta = p[0] * zeta0_ev
        gs = p[1] * g_s0 if fit_g_s else g_s0
        return zeta, gs

    def residuals(p):
        zeta, gs = unpack(p)
        return ((energy_vs_field(l, b_values, zeta, g_s=gs) - measured) / zeta0_ev).ravel()

    p0 = np.array([1.0, 1.0]) if fit_g_s else np.array([1.0])
    sol = least_squares(residuals, p0, xtol=1e-12, ftol=1e-12)
    zeta, gs = unpack(sol.x)
    rms = float(np.sqrt(np.mean((residuals(sol.x) * zeta0_ev) ** 2)))
    return FitResult(zeta_ev=float(zeta), g_s=float(gs), rms_residual_ev=rms, success=bool(sol.success))
