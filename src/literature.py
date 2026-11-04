r"""Cross-check against a published calculation: sodium D-line groups at 0.9 T.

Source: "Sub-Doppler spectra of sodium D lines in a wide range of magnetic field: theoretical study", arXiv:2104.14896.
That paper includes hyperfine structure (nuclear spin 3/2); at 0.9 T the field is about 14 times the hyperfine field
B0 = A_hf/mu_B = 63.3 mT, so the lines gather into GROUPS (one per fine-structure Zeeman component) whose spacings are
set by the fine-structure Zeeman effect alone. Our model has no hyperfine structure, so the group spacings are what we
can compare. Honest limits: (1) the paper is a THEORY study, not laboratory data; (2) some of its numbers are
read off its figures; (3) hyperfine structure shifts and broadens each group by a few hundred MHz, so agreement to
about 0.5 % of a 17 to 42 GHz spacing is the expected level, not an exact match.
Values below were quoted from the paper's text (Eqs. 16, 21, 23, 25 and 26 and the sentences around them); check
them against the paper before citing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import constants as sc

from .constants import wavenumber_to_ev
from .transitions import zeeman_lines
from .validation import zeta_from_fine_structure

MU_B_MHZ_PER_T = 1.39962449361e4        # Table I of the paper (Hz/T / 1e6)
B_TEST_T = 0.9
D1_CM1, FINE_CM1 = 16956.17, 17.196

# group, formula (in units of mu_B B), paper formula value (GHz), paper numerical or figure value (GHz), how it was obtained
PAPER_AT_0P9T = pd.DataFrame([
    ("D1 pi",     "4/3", 16.8,  16.9,  "numerical calculation"),
    ("D1 sigma",  "8/3", 33.59, 33.6,  "read from the paper's figure"),
    ("D2 pi",     "2/3", 8.40,  8.46,  "numerical calculation"),
    ("D2 sigma r1", "2",   25.19, 25.15, "read from the paper's figure"),
    ("D2 sigma r2", "10/3", 41.98, 42.07, "read from the paper's figure"),
], columns=["group", "formula_mu_b_B", "paper_formula_ghz", "paper_numerical_ghz", "paper_numerical_how"])


def formula_ghz(factor: float, b_tesla: float = B_TEST_T) -> float:
    """Group spacing factor x mu_B B in GHz (mu_B from the paper's Table I)."""
    return factor * MU_B_MHZ_PER_T * b_tesla / 1e3


def model_group_spacings_ghz(b_tesla: float = B_TEST_T) -> dict[str, float]:
    """Spacings (GHz) between the line groups of our fine-structure-only model at field B."""
    zeta = zeta_from_fine_structure(FINE_CM1, l=1)
    gap = wavenumber_to_ev(D1_CM1) + zeta
    df = zeeman_lines(1, 0, b_tesla, zeta, 0.0, gap)
    df["nu_ghz"] = df.wavenumber_cm1 * sc.c * 100 / 1e9
    cut = D1_CM1 + FINE_CM1 / 2
    d1, d2 = df[df.wavenumber_cm1 < cut], df[df.wavenumber_cm1 >= cut]
    pi = lambda d: np.sort(d[d.polarisation == "pi"].nu_ghz.values)
    sp = lambda d: np.sort(d[d.polarisation == "sigma+"].nu_ghz.values)
    sm = lambda d: np.sort(d[d.polarisation == "sigma-"].nu_ghz.values)
    out = {"D1 pi": pi(d1)[-1] - pi(d1)[0], "D1 sigma": sp(d1)[0] - sm(d1)[0]}
    out["D2 pi"] = pi(d2)[-1] - pi(d2)[0]
    # sigma+ lines sit at +1 and +5/3 and sigma- at -1 and -5/3 (units of mu_B B): inner pair and outer pair
    out["D2 sigma r1"] = sp(d2)[0] - sm(d2)[-1]
    out["D2 sigma r2"] = sp(d2)[-1] - sm(d2)[0]
    return out


def comparison_table(b_tesla: float = B_TEST_T) -> pd.DataFrame:
    """Model against the paper's formula and numerical values, with percentage differences."""
    model = model_group_spacings_ghz(b_tesla)
    t = PAPER_AT_0P9T.copy()
    t["model_ghz"] = t.group.map(model)
    t["diff_vs_formula_%"] = 100 * (t.model_ghz / t.paper_formula_ghz - 1)
    t["diff_vs_numerical_%"] = 100 * (t.model_ghz / t.paper_numerical_ghz - 1)
    return t


def fig_literature_comparison(b_tesla: float = B_TEST_T):
    """Percentage difference of our model from the paper's formula and from its numerical / figure values, per group."""
    import matplotlib.pyplot as plt
    t = comparison_table(b_tesla)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-1, 1, color="0.92", zorder=0)
    ax.scatter(t["diff_vs_formula_%"], y + 0.12, color="tab:blue", label="vs the paper's formula (μ_B B × factor)", zorder=3)
    ax.scatter(t["diff_vs_numerical_%"], y - 0.12, color="tab:orange", label="vs the paper's numerical or figure value", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  ({m} μ_B B)" for g, m in zip(t.group, t.formula_mu_b_B)])
    ax.invert_yaxis()
    ax.set_xlim(-1.2, 1.2)
    ax.set_xlabel("difference of our model from the published value (%)   (grey band: ±1 %)")
    ax.set_title(f"Sodium D-line group spacings at {b_tesla:g} T against arXiv:2104.14896 (published theory, with hyperfine)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig
