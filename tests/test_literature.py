"""Tests for src/literature.py: cross-check against the published sodium D-line group spacings at 0.9 T."""
import pytest

from src.literature import (MU_B_MHZ_PER_T, PAPER_AT_0P9T, comparison_table, fig_literature_comparison, formula_ghz,
                            model_group_spacings_ghz)


def test_paper_values_reproduce_their_own_formulas():
    # sanity of the typed-in reference table: the paper's formula column equals factor x mu_B x 0.9 T (to its rounding)
    for _, r in PAPER_AT_0P9T.iterrows():
        num, den = (r.formula_mu_b_B.split("/") + ["1"])[:2]
        assert formula_ghz(float(num) / float(den)) == pytest.approx(r.paper_formula_ghz, abs=0.02)


def test_model_agrees_with_paper_numerical_values_within_one_percent():
    t = comparison_table()
    assert (t["diff_vs_numerical_%"].abs() < 1.0).all()


def test_model_agrees_with_paper_formulas_within_half_a_percent():
    assert (comparison_table()["diff_vs_formula_%"].abs() < 0.5).all()


def test_spacings_scale_with_field():
    a, b = model_group_spacings_ghz(0.45), model_group_spacings_ghz(0.9)
    for k in a:
        assert b[k] / a[k] == pytest.approx(2.0, rel=0.01)


def test_bohr_magneton_matches_scipy():
    from scipy import constants as sc
    assert MU_B_MHZ_PER_T == pytest.approx(sc.physical_constants["Bohr magneton in Hz/T"][0] / 1e6, rel=1e-8)


def test_figure_builds():
    import matplotlib
    matplotlib.use("Agg")
    assert fig_literature_comparison() is not None
