r"""The global uncertainty budget: one table for every system in the project.

Each row is an observable of one system (sodium, hydrogen n = 2, H-alpha, calcium), with

* a model function and its inputs ``{name: (mean, sigma)}``,
* where each sigma comes from: ``measured`` (a published uncertainty) or ``assumed`` (my illustrative choice; replace it
  when a real value is found),
* optionally a reference value with its own sigma, so the row gets a z-score.

``budget_table()`` runs the Monte Carlo for all rows and returns one DataFrame; ``fig_budget()`` draws it.
Calcium has no hyperfine row because Ca-40 has no nuclear spin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .uncertainty import (hydrogen_crossing_gauss, monte_carlo, sigma_compare, sodium_3p12_splitting_mhz)


@dataclass
class Row:
    system: str
    observable: str
    unit: str
    model: callable
    inputs: dict
    sigma_source: str                       # "measured" or "assumed"
    reference: float | None = None
    ref_sigma: float = 0.0
    note: str = ""


def _ca_admixture(e3p0, e3p1, e3p2, e1p1):
    from .two_electron import fit_to_levels, singlet_admixture_in_3p1
    z, d, et, _ = fit_to_levels(e3p0, e3p1, e3p2, e1p1)
    return singlet_admixture_in_3p1(z, d, et)


def _ca_interval_ratio(e3p0, e3p1, e3p2, e1p1):
    return (e3p2 - e3p1) / (e3p1 - e3p0)


def _halpha_doppler_pm(temperature_k):
    from .hydrogen_spectrum import doppler_fwhm_pm
    return doppler_fwhm_pm(temperature_k=temperature_k)


def rows() -> list[Row]:
    from .two_electron import CA_LEVELS_CM1
    ca = {k: (v, 0.001) for k, v in zip(("e3p0", "e3p1", "e3p2", "e1p1"),
                                       (CA_LEVELS_CM1["3P0"], CA_LEVELS_CM1["3P1"], CA_LEVELS_CM1["3P2"], CA_LEVELS_CM1["1P1"]))}
    return [
        Row("Sodium", "3P1/2 hyperfine splitting", "MHz", sodium_3p12_splitting_mhz, {"a_3p12": (94.44, 0.13)},
            "measured", note="A(3P1/2) = 94.44(13) MHz, Steck's table; splitting = 2A"),
        Row("Hydrogen n=2", "2S x 2P crossing (no hyperfine)", "G", hydrogen_crossing_gauss,
            {"lamb_mhz": (1057.845, 0.009)}, "measured",
            note="Lamb shift 1057.845(9) MHz, Lundeen and Pipkin as quoted in arXiv:0809.2985"),
        Row("Hydrogen H-alpha", "Doppler FWHM at 300 K", "pm", _halpha_doppler_pm, {"temperature_k": (300.0, 10.0)},
            "assumed", note="temperature known to ±10 K is an assumption about the lab"),
        Row("Calcium 4s4p", "Landé interval ratio", "-", _ca_interval_ratio, ca, "assumed",
            note="NIST Handbook energies; sigma 0.001 cm^-1 is only the tabulated precision (assumed); the rule value 2 is an approximation, so no z-score"),
        Row("Calcium 4s4p", "singlet admixture in 3P1", "-", _ca_admixture, ca, "assumed",
            note="NIST Handbook energies; sigma 0.001 cm^-1 is only the tabulated precision (assumed)"),
    ]


def g_factor_rows() -> pd.DataFrame:
    """Sodium g_J: model (no free inputs, sigma 0) against the measurements, with z-scores."""
    from .literature import g_factor_comparison
    t = g_factor_comparison()
    out = []
    for _, r in t.iterrows():
        z = sigma_compare(r.model, 0.0, r.measured, r.uncertainty)
        out.append({"system": "Sodium", "observable": f"g_J {r.level}", "unit": "-", "value": r.model, "sigma": 0.0,
                    "sigma_source": "measured", "reference": r.measured, "ref_sigma": r.uncertainty,
                    "z": z["z"], "verdict": z["verdict"], "note": "model has no free inputs (uses the free-electron g_s); reference is the measurement"})
    return pd.DataFrame(out)


def budget_table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    out = []
    for r in rows():
        mc = monte_carlo(r.model, r.inputs, n=n, seed=seed)
        z = sigma_compare(mc["mean"], mc["std"], r.reference, r.ref_sigma) if r.reference is not None else {"z": np.nan, "verdict": ""}
        out.append({"system": r.system, "observable": r.observable, "unit": r.unit, "value": mc["mean"],
                    "sigma": mc["std"], "sigma_source": r.sigma_source, "reference": r.reference if r.reference is not None else np.nan,
                    "ref_sigma": r.ref_sigma, "z": z["z"], "verdict": z["verdict"], "note": r.note})
    return pd.concat([pd.DataFrame(out), g_factor_rows()], ignore_index=True)


def fig_budget(table: pd.DataFrame | None = None):
    """Relative uncertainty of every observable (log scale); colour = where sigma comes from."""
    t = budget_table() if table is None else table
    t = t[t.sigma > 0].copy()
    t["rel"] = (t.sigma / t.value.abs()).clip(lower=1e-12)
    t = t.sort_values("rel")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(t) + 1.6))
    colors = ["#1b7837" if s == "measured" else "#e08214" for s in t.sigma_source]
    ax.barh([f"{a}: {b}" for a, b in zip(t.system, t.observable)], t.rel * 100, color=colors)
    ax.set_xscale("log"); ax.set_xlabel("relative uncertainty (%)")
    ax.set_title("Global uncertainty budget (green: published sigma, orange: assumed sigma)")
    fig.tight_layout()
    return fig
