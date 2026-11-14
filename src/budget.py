r"""The global uncertainty budget: one table for every system in the project.

Each row is an observable of one system (sodium, hydrogen n = 2, H-alpha, calcium), with

* a model function and its inputs ``{name: (mean, sigma)}``,
* where each sigma comes from: ``measured`` (a published uncertainty) or ``assumed`` (an illustrative choice or SCENARIO;
  replace it when a real value is found). The calcium rows use a scenario sigma of 0.001 cm^-1 per level
  (``CA_SCENARIO_SIGMA_CM1``): that is the rounding of the NIST Handbook table, which lists NO uncertainties, so the
  resulting sigma is "what would follow if every level were uncertain by that much", not a measured standard error,
* optionally a reference value with its own sigma, so the row gets a z-score.

``budget_table()`` runs the Monte Carlo for all rows and returns one DataFrame; ``fig_budget()`` draws it.

Columns of the table (three different kinds of "error" are kept apart):

* ``value``: the Monte Carlo MEAN of the output (unchanged contract; it carries sampling noise of order sigma/sqrt(n));
* ``nominal``: the DETERMINISTIC model value at the central inputs (use this as the best estimate of the observable);
* ``sigma``: the PROPAGATED output spread (standard deviation of the output for the stated input uncertainties);
* ``mc_sem``: Monte Carlo sampling error of the mean (named ``mc_sem``, not ``sem``, because ``DataFrame.sem`` is a pandas method), sigma/sqrt(n). It is a property of the sampling, not of the physics;
* ``n_samples`` and ``sigma_rel_sampling``: the number of draws and the approximate relative sampling error of ``sigma``
  itself, 1/sqrt(2(n-1)) (assumes a near-normal output). Neither ``mc_sem`` nor ``sigma_rel_sampling`` is an input or model
  uncertainty. Model errors (omitted interactions) are not in any of these columns.
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


# Scenario sigma for the calcium levels: the NIST Handbook (SC85) lists no uncertainties; 0.001 cm^-1 is only the rounding
# of the table. Other scenarios (0.01, 0.02 cm^-1) are used in tests/test_calcium_audit.py and docs/ext_two_electron.tex.
CA_SCENARIO_SIGMA_CM1 = 0.001


def _ca_admixture(e3p0, e3p1, e3p2, e1p1):
    """Singlet admixture eps^2 of 3P1 from the closed-form FOUR-parameter fit (zeta_S, zeta_A, Delta, E_t),
    eps^2 = (a - E(3P1)) / (E(1P1) - E(3P1)) with a = (E(3P2) + 2 E(3P0)) / 3. This is a calibration of four numbers by four
    parameters; the propagated sigma is a sensitivity SCENARIO for assumed level uncertainties."""
    from .two_electron import calcium_four_parameter_fit
    return calcium_four_parameter_fit(e3p0, e3p1, e3p2, e1p1)["eps2"]


def _ca_admixture_constrained(e3p0, e3p1, e3p2, e1p1):
    """eps^2 from the OLD three-parameter model with zeta_A tied to zeta_S (term zeta L.s2). It misses the 3P1 level by 0.089 cm^-1
    and is NOT used in the budget; it is kept only to show why the earlier number (7.8e-5) differs from the four-parameter one
    (6.1e-5): eps^2 scales as zeta_A^2 and the tie forces zeta_A = 52.7 instead of the fitted 46.75 cm^-1."""
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
    ca = {k: (v, CA_SCENARIO_SIGMA_CM1) for k, v in zip(("e3p0", "e3p1", "e3p2", "e1p1"),
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
            note="NIST Handbook energies; SCENARIO sigma 0.001 cm^-1 per level (table rounding, not a published uncertainty); the rule value 2 is an approximation, so no z-score"),
        Row("Calcium 4s4p", "singlet admixture in 3P1", "-", _ca_admixture, ca, "assumed",
            note="four-parameter closed form (calibration); SCENARIO sigma 0.001 cm^-1 per level (table rounding, NIST SC85 lists no uncertainties); not a standard error; omitted perturbers are model uncertainty and are not included"),
    ]


def g_factor_rows() -> pd.DataFrame:
    """Sodium g_J: model (no free inputs, sigma 0) against the measurements, with z-scores."""
    from .literature import g_factor_comparison
    t = g_factor_comparison()
    out = []
    for _, r in t.iterrows():
        z = sigma_compare(r.model, 0.0, r.measured, r.uncertainty)
        out.append({"system": "Sodium", "observable": f"g_J {r.level}", "unit": "-", "value": r.model, "nominal": r.model, "sigma": 0.0,
                    "mc_sem": np.nan, "n_samples": 0, "sigma_rel_sampling": np.nan,
                    "sigma_source": "measured", "reference": r.measured, "ref_sigma": r.uncertainty,
                    "z": z["z"], "verdict": z["verdict"], "note": "model has no free inputs (uses the free-electron g_s); reference is the measurement"})
    return pd.DataFrame(out)


def budget_table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """One row per observable. ``value`` is the Monte Carlo mean (unchanged); ``nominal`` is the deterministic central value;
    ``sigma`` the propagated spread; ``mc_sem`` the sampling error of the MC mean; see the module docstring."""
    out = []
    for r in rows():
        mc = monte_carlo(r.model, r.inputs, n=n, seed=seed)
        nominal = float(r.model(**{k: m for k, (m, _) in r.inputs.items()}))
        z = sigma_compare(mc["mean"], mc["std"], r.reference, r.ref_sigma) if r.reference is not None else {"z": np.nan, "verdict": ""}
        out.append({"system": r.system, "observable": r.observable, "unit": r.unit, "value": mc["mean"], "nominal": nominal,
                    "sigma": mc["std"], "mc_sem": mc["sem"], "n_samples": n, "sigma_rel_sampling": 1.0 / np.sqrt(2.0 * (n - 1)),
                    "sigma_source": r.sigma_source, "reference": r.reference if r.reference is not None else np.nan,
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
