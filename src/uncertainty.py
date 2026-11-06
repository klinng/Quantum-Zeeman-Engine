r"""Programmable uncertainty: how far can we trust a number the model gives?

Three small tools.

1. ``monte_carlo(model, inputs, n)``: draw every input from a normal distribution
   (mean, sigma), run the model for each draw, and report mean and standard deviation of the output.
   This is "propagation by sampling": no algebra needed, works for any model including eigenvalue problems.
2. ``linear_sensitivity(model, inputs)``: the same result by derivatives, sigma_out^2 = sum (df/dx_i * sigma_i)^2.
   Useful to see WHICH input dominates, and as a cross-check of the sampling.
3. ``sigma_compare(a, sig_a, b, sig_b)``: how many combined standard deviations apart are two numbers?
   z < 2 means "consistent", z > 3 means "a real disagreement".

Inputs are a dict ``{name: (mean, sigma)}``; the model is a function ``f(**values) -> float``.
Samples are independent (no correlations); that is stated, not hidden.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .hydrogen import FINE_2P_MHZ, LAMB_SHIFT_MHZ, h_n2


def monte_carlo(model, inputs: dict, n: int = 400, seed: int = 0) -> dict:
    """Propagate input uncertainties through ``model`` by random sampling.

    Returns dict(samples, mean, std, sem) where sem = std/sqrt(n) is the error of the mean itself.
    """
    rng = np.random.default_rng(seed)
    draws = {k: rng.normal(m, s, n) for k, (m, s) in inputs.items()}
    out = np.array([model(**{k: v[i] for k, v in draws.items()}) for i in range(n)], dtype=float)
    return {"samples": out, "mean": float(out.mean()), "std": float(out.std(ddof=1)),
            "sem": float(out.std(ddof=1) / np.sqrt(n)), "draws": draws}


def linear_sensitivity(model, inputs: dict, rel_step: float = 1e-4) -> dict:
    """Central-difference derivatives. Returns dict(value, derivs, contributions, std)."""
    base = {k: m for k, (m, _) in inputs.items()}
    derivs, contrib = {}, {}
    for k, (m, s) in inputs.items():
        h = abs(m) * rel_step if m != 0 else rel_step
        up, dn = dict(base), dict(base)
        up[k], dn[k] = m + h, m - h
        derivs[k] = (model(**up) - model(**dn)) / (2 * h)
        contrib[k] = abs(derivs[k]) * s
    return {"value": float(model(**base)), "derivs": derivs, "contributions": contrib,
            "std": float(np.sqrt(sum(c ** 2 for c in contrib.values())))}


def sigma_compare(a: float, sig_a: float, b: float, sig_b: float = 0.0) -> dict:
    """z = |a - b| / sqrt(sig_a^2 + sig_b^2) and a plain-words verdict."""
    s = float(np.hypot(sig_a, sig_b))
    if s == 0:
        z = 0.0 if a == b else float("inf")
    else:
        z = abs(a - b) / s
    verdict = "consistent" if z < 2 else ("borderline" if z < 3 else "disagreement")
    return {"z": z, "verdict": verdict}


# ---- ready-made models ---------------------------------------------------------------------------
def hydrogen_crossing_gauss(lamb_mhz: float = LAMB_SHIFT_MHZ, fine_mhz: float = FINE_2P_MHZ) -> float:
    """Field (gauss) where 2S1/2 (m_j = -1/2) crosses 2P1/2 (m_j = +1/2), no hyperfine.

    The 2s and 2p blocks never mix in a magnetic field, so we solve each block separately and find the root.
    """
    def gap(b):
        h = h_n2(b, fine_mhz=fine_mhz, lamb_mhz=lamb_mhz)
        e_s = np.linalg.eigvalsh(h[:2, :2])[0]           # 2S1/2 with m_j = -1/2 (falling)
        e_p = np.linalg.eigvalsh(h[2:, 2:])[1]           # 2P1/2 with m_j = +1/2 (second lowest)
        return e_s - e_p
    return brentq(gap, 0.050, 0.065, xtol=1e-9) * 1e4


def sodium_3p12_splitting_mhz(a_3p12: float) -> float:
    """Hyperfine splitting of Na 3P1/2 (F = 2 minus F = 1) = 2 A for I = 3/2, from the full matrix model."""
    from .hydrogen import ev_to_mhz
    from .hyperfine import NA_A_MHZ, NA_G_I, NA_I, h_with_hyperfine
    from .hyperfine import mhz_to_ev
    a = {0.5: mhz_to_ev(a_3p12), 1.5: mhz_to_ev(NA_A_MHZ["3P3/2"])}
    zeta = 0.002
    w = np.sort(np.linalg.eigvalsh(h_with_hyperfine(1, NA_I, a, 0.0, NA_G_I, zeta, e0_ev=zeta)))
    return float(ev_to_mhz(w[:8].max() - w[:8].min()))
