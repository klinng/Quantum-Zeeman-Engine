r"""Differentiable line profiles and Zeeman spectrum: Voigt profile from the Faddeeva function, in plain arithmetic.

``scipy.special.voigt_profile`` is not differentiable in PyTorch, and PyTorch has no complex error function. The Voigt
profile is, however, V(x; sigma, gamma) = Re w(z) / (sigma sqrt(2 pi)),  z = (x + i gamma) / (sigma sqrt 2),
with w the Faddeeva function w(z) = exp(-z^2) erfc(-i z). Here w is computed with the rational-function approximation of
Weideman (SIAM J. Numer. Anal. 31, 1497, 1994): for Im z >= 0

    w(z) = 2 p(Z) / (L - i z)^2 + (1/sqrt(pi)) / (L - i z),   Z = (L + i z) / (L - i z),   L = sqrt(N / sqrt 2),

where p is a polynomial of degree N-1 whose coefficients come from one FFT (computed once with NumPy, constants).
Evaluating p by Horner's rule uses only +, * and /, so the SAME function works on NumPy arrays and on PyTorch tensors,
and PyTorch differentiates it automatically with respect to x, sigma and gamma (hence B, T and P).

Physical parameters (all inputs of ``zeeman_spectrum``):
* Doppler  sigma = nu0 sqrt(k_B T / (m c^2))              (Gaussian standard deviation, Hz)
* Lorentz  gamma = A / (4 pi) + k_P P                      (half width at half maximum, Hz); k_P is the pressure-broadening
  coefficient in Hz/Pa, an INPUT (no collision coefficient is built in).
* Line positions and strengths come from the eigenvalues and eigenvectors of the Zeeman Hamiltonians of the upper (l = 1,
  spin-orbit zeta) and lower (l = 0) manifolds, the same model as src/transitions.py.
Gradients of eigenvectors need non-degenerate levels: use B > 0 (at B = 0 the levels are degenerate and the gradient is not defined).
"""
from __future__ import annotations

import numpy as np

SQRT_PI = np.sqrt(np.pi)
K_B = 1.380649e-23          # J/K (exact, SI 2019)
C = 299_792_458.0           # m/s (exact)
AMU = 1.66053906660e-27     # kg (CODATA 2018)
H_PLANCK = 6.62607015e-34   # J s (exact)
EV = 1.602176634e-19        # J (exact)


def weideman_coefficients(n: int = 32):
    """Polynomial coefficients a_0..a_{N-1} (highest power first) and the scale L of the Weideman Faddeeva approximation."""
    m = 2 * n
    m2 = 2 * m
    length = np.sqrt(n / np.sqrt(2.0))
    k = np.arange(-m + 1, m)
    theta = k * np.pi / m
    t = length * np.tan(theta / 2)
    f = np.exp(-t ** 2) * (length ** 2 + t ** 2)
    f = np.concatenate(([0.0], f))
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / m2
    return a[1:n + 1][::-1].tolist(), float(length)


_COEF, _L = weideman_coefficients(32)


def faddeeva(z):
    """Faddeeva function w(z) for Im z >= 0, on NumPy arrays or PyTorch tensors (differentiable). Accuracy ~1e-12 for N = 32."""
    denom = _L - 1j * z
    big_z = (_L + 1j * z) / denom
    p = _COEF[0]
    for c in _COEF[1:]:
        p = p * big_z + c
    return 2 * p / denom ** 2 + (1.0 / SQRT_PI) / denom


def voigt(x, sigma, gamma):
    """Area-1 Voigt profile of a Gaussian (standard deviation sigma) convolved with a Lorentzian (HWHM gamma).
    Works with floats, NumPy arrays and PyTorch tensors; sigma > 0, gamma >= 0."""
    z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
    return faddeeva(z).real / (sigma * np.sqrt(2 * np.pi))


def doppler_sigma_hz(nu0_hz, temperature_k, mass_amu):
    """Gaussian standard deviation (Hz) of the Doppler-broadened line: nu0 sqrt(k_B T / (m c^2)). Differentiable in T."""
    return nu0_hz * (K_B * temperature_k / (mass_amu * AMU * C ** 2)) ** 0.5


def lorentz_hwhm_hz(a_total_per_s, pressure_pa, k_pressure_hz_per_pa):
    """Lorentzian HWHM (Hz): natural A/(4 pi) plus pressure k_P * P. Differentiable in P."""
    return a_total_per_s / (4 * np.pi) + k_pressure_hz_per_pa * pressure_pa


def _problem(zeta_ev: float, g_s: float):
    """Constant (NumPy) operators of the l = 1 upper and l = 0 lower manifolds and the dipole matrices."""
    from .hamiltonians import coupled_ops
    from .transitions import dipole_matrices
    up, low = coupled_ops(1), coupled_ops(0)
    ls = up["Lx"] @ up["Sx"] + up["Ly"] @ up["Sy"] + up["Lz"] @ up["Sz"]
    return up, low, ls, dipole_matrices(1, 0)


def line_list(b_tesla, zeta_ev: float, backend: str = "numpy", g_l: float = 1.0, g_s: float = 2.00231930436):
    """Transition frequencies (Hz, from the zero-field centroid) and strengths of the l = 1 -> l = 0 Zeeman lines.

    backend 'numpy' returns arrays; backend 'torch' returns tensors and keeps the graph to b_tesla (a tensor).
    Strengths are |<up|D_q|low>|^2 summed over the three polarisations, normalised so the sum is the number of upper states."""
    up, low, ls, dmat = _problem(zeta_ev, g_s)
    mu_b_ev = 5.7883818060e-5
    if backend == "torch":
        import torch
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.complex128)
        h_up = zeta_ev * t(ls) + mu_b_ev * b_tesla * (g_l * t(up["Lz"]) + g_s * t(up["Sz"]))
        h_low = mu_b_ev * b_tesla * g_s * t(low["Sz"])
        eu, vu = torch.linalg.eigh(h_up)
        el, vl = torch.linalg.eigh(h_low)
        strength = sum((vu.conj().T @ t(d) @ vl).abs() ** 2 for d in dmat.values())
        strength = strength * (eu.shape[0] / strength.sum())
        de = (eu[:, None] - el[None, :]) / (H_PLANCK / EV)
        return de.reshape(-1), strength.reshape(-1)
    h_up = zeta_ev * ls + mu_b_ev * b_tesla * (g_l * up["Lz"] + g_s * up["Sz"])
    h_low = mu_b_ev * b_tesla * g_s * low["Sz"]
    eu, vu = np.linalg.eigh(h_up)
    el, vl = np.linalg.eigh(h_low)
    strength = sum(np.abs(vu.conj().T @ d @ vl) ** 2 for d in dmat.values())
    strength = strength * (len(eu) / strength.sum())
    de = (eu[:, None] - el[None, :]) / (H_PLANCK / EV)
    return de.reshape(-1), strength.reshape(-1)


def zeeman_spectrum(nu_hz, b_tesla, temperature_k, pressure_pa, zeta_ev: float, nu0_hz: float, mass_amu: float,
                    a_total_per_s: float, k_pressure_hz_per_pa: float, backend: str = "numpy"):
    """Spectrum S(nu) = sum_lines strength * Voigt(nu - nu_line; sigma(T), gamma(P)), fully differentiable with backend='torch'
    with respect to b_tesla, temperature_k and pressure_pa (pass them as tensors with requires_grad=True)."""
    freqs, strengths = line_list(b_tesla, zeta_ev, backend)
    sigma = doppler_sigma_hz(nu0_hz, temperature_k, mass_amu)
    gamma = lorentz_hwhm_hz(a_total_per_s, pressure_pa, k_pressure_hz_per_pa)
    if backend == "torch":
        x = nu_hz[:, None] - freqs[None, :]
        return (strengths[None, :] * voigt(x, sigma, gamma)).sum(dim=1)
    x = np.asarray(nu_hz)[:, None] - freqs[None, :]
    return (strengths[None, :] * voigt(x, sigma, gamma)).sum(axis=1)
