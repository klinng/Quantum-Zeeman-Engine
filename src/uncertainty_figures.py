"""Figures that explain the uncertainty tools (src/uncertainty.py)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .uncertainty import (hydrogen_crossing_gauss, linear_sensitivity, monte_carlo,
                          sodium_3p12_splitting_mhz)


def _hist(ax, samples, mean, std, lin_std, xlabel, title):
    ax.hist(samples, bins=25, density=True, color="#9ecae1", edgecolor="white")
    x = np.linspace(samples.min(), samples.max(), 200)
    ax.plot(x, np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi)), "k", lw=1.5,
            label=f"sampling: {mean:.3f} ± {std:.3f}")
    ax.axvspan(mean - lin_std, mean + lin_std, color="tab:red", alpha=0.15, label=f"derivatives: ± {lin_std:.3f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("probability density"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def fig_uncertainty(sigma_lamb_mhz: float = 0.009, n: int = 400):
    """Left: hydrogen 573 G crossing with the Lamb shift 1057.845(9) MHz (Lundeen and Pipkin, as quoted in arXiv:0809.2985).
    Middle: sodium 3P1/2 hyperfine splitting from A(3P1/2) = 94.44(13) MHz (Steck). Right: how the
    sampling error shrinks with the number of samples (1/sqrt(2N) for a standard deviation)."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
    inp = {"lamb_mhz": (1057.845, sigma_lamb_mhz)}
    r = monte_carlo(hydrogen_crossing_gauss, inp, n)
    l = linear_sensitivity(hydrogen_crossing_gauss, inp)
    _hist(a1, r["samples"], r["mean"], r["std"], l["std"], "crossing field (G)",
          f"H 2S×2P crossing, Lamb shift ± {sigma_lamb_mhz:g} MHz")
    inp2 = {"a_3p12": (94.44, 0.13)}
    r2 = monte_carlo(sodium_3p12_splitting_mhz, inp2, n)
    l2 = linear_sensitivity(sodium_3p12_splitting_mhz, inp2)
    _hist(a2, r2["samples"], r2["mean"], r2["std"], l2["std"], "3P1/2 hyperfine splitting (MHz)",
          "Na 3P1/2 splitting = 2A, A = 94.44(13) MHz")
    sizes = np.unique(np.logspace(1, 3.3, 14).astype(int))
    stds = [monte_carlo(sodium_3p12_splitting_mhz, inp2, int(k), seed=3)["std"] for k in sizes]
    a3.semilogx(sizes, stds, "o-", label="sampled std")
    a3.axhline(0.26, color="k", ls="--", label="exact 2σ_A = 0.26")
    a3.fill_between(sizes, 0.26 * (1 - 1 / np.sqrt(2 * sizes)), 0.26 * (1 + 1 / np.sqrt(2 * sizes)),
                    color="gray", alpha=0.2, label="expected scatter 1/√(2N)")
    a3.set_xlabel("number of samples N"); a3.set_ylabel("sampled σ (MHz)")
    a3.set_title("More samples, tighter estimate", fontsize=10); a3.legend(fontsize=8)
    fig.tight_layout()
    return fig
