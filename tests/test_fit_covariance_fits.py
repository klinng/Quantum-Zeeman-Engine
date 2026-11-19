"""Stage 8B.2a: parameter covariance of a fit from a Jacobian (``fit_parameter_covariance``) and per-element provenance.

Every test is deterministic and uses synthetic, analytically solvable problems. They prove that the formulas, the ordering contract
and the refusal/unavailability paths are implemented as documented. They do NOT show that any physical model or any assumed
uncertainty is right: no fitting routine calls this function yet, and nothing here is an experimental validation."""
import numpy as np
import pytest

from src.fit_covariance import (CovarianceError, FitCovariance, MeasurementSet, fit_parameter_covariance, numerical_jacobian,
                                inverse_map_covariance)

X = np.column_stack([np.ones(6), np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])])      # straight line a + b x
NAMES = ("a", "b")


def _ms(sig, kind="scenario", n=6, cov=None):
    names = [f"y{i}" for i in range(n)]
    if cov is not None:
        return MeasurementSet.from_covariance(names, np.zeros(n), cov, kind, "synthetic test covariance")
    return MeasurementSet.independent(names, np.zeros(n), sig, kind, "synthetic test covariance")


# ------------------------------------------------------------------------------------------------ weighted
def test_weighted_matches_closed_form_for_a_straight_line_with_known_sigmas():
    sig = np.array([1.0, 2.0, 0.5, 1.0, 4.0, 1.5])
    w = 1 / sig ** 2
    res = fit_parameter_covariance(X, mode="weighted", cov_y=_ms(sig), objective_is_weighted=True, param_names=NAMES)
    sw, sx, sxx = w.sum(), (w * X[:, 1]).sum(), (w * X[:, 1] ** 2).sum()
    det = sw * sxx - sx ** 2
    expect = np.array([[sxx, -sx], [-sx, sw]]) / det                        # textbook weighted straight-line covariance
    assert res.kind == "weighted" and res.status == "ok" and res.available
    assert np.allclose(res.covariance, expect, rtol=1e-12)
    assert np.allclose(res.covariance, np.linalg.inv(X.T @ np.diag(w) @ X), rtol=1e-12)
    assert np.allclose(res.std, np.sqrt(np.diag(expect)))


def test_weighted_mode_requires_the_objective_to_be_declared_weighted_and_a_positive_definite_covariance():
    with pytest.raises(ValueError, match="objective_is_weighted"):
        fit_parameter_covariance(X, mode="weighted", cov_y=_ms(1.0))
    singular = np.ones((6, 6))
    with pytest.raises(CovarianceError, match="positive-definite"):
        fit_parameter_covariance(X, mode="weighted", cov_y=singular, objective_is_weighted=True)


def test_weighted_reports_chi2_but_never_rescales_the_covariance():
    rng_res = np.array([0.5, -1.0, 2.0, -0.5, 1.0, -2.0])
    sig = np.full(6, 2.0)
    base = fit_parameter_covariance(X, mode="weighted", cov_y=_ms(sig), objective_is_weighted=True)
    withr = fit_parameter_covariance(X, mode="weighted", cov_y=_ms(sig), objective_is_weighted=True, residuals=rng_res)
    assert np.array_equal(base.covariance, withr.covariance)                 # chi^2 is reported, not used
    assert withr.chi2 == pytest.approx(np.sum((rng_res / sig) ** 2)) and withr.dof == 4
    assert withr.chi2_per_dof == pytest.approx(withr.chi2 / 4)


# ------------------------------------------------------------------------------------------------ sandwich and correlation
def test_sandwich_matches_the_closed_form_for_an_unweighted_estimator_with_heteroscedastic_noise():
    sig = np.array([1.0, 2.0, 0.5, 1.0, 4.0, 1.5])
    c = np.diag(sig ** 2)
    res = fit_parameter_covariance(X, mode="sandwich", cov_y=_ms(sig))
    inv = np.linalg.inv(X.T @ X)
    assert np.allclose(res.covariance, inv @ X.T @ c @ X @ inv, rtol=1e-12)
    wt = fit_parameter_covariance(X, mode="weighted", cov_y=_ms(sig), objective_is_weighted=True)
    assert not np.allclose(res.covariance, wt.covariance, rtol=1e-3)         # a different estimator, a different covariance


def test_sandwich_equals_weighted_and_ols_formula_when_errors_are_iid():
    s = 0.7
    sw = fit_parameter_covariance(X, mode="sandwich", cov_y=_ms(s))
    wt = fit_parameter_covariance(X, mode="weighted", cov_y=_ms(s), objective_is_weighted=True)
    assert np.allclose(sw.covariance, wt.covariance, rtol=1e-12)
    assert np.allclose(sw.covariance, s ** 2 * np.linalg.inv(X.T @ X), rtol=1e-12)


def test_correlated_errors_use_the_full_covariance_and_obey_gauss_markov():
    n = 6
    idx = np.arange(n)
    cov = 0.5 ** np.abs(idx[:, None] - idx[None, :]) * 1.7                     # AR(1)-type correlation, positive definite
    ms = _ms(None, cov=cov)
    gls = fit_parameter_covariance(X, mode="weighted", cov_y=ms, objective_is_weighted=True)
    sw = fit_parameter_covariance(X, mode="sandwich", cov_y=ms)
    inv = np.linalg.inv(X.T @ X)
    assert np.allclose(gls.covariance, np.linalg.inv(X.T @ np.linalg.inv(cov) @ X), rtol=1e-10)
    assert np.allclose(sw.covariance, inv @ X.T @ cov @ X @ inv, rtol=1e-10)
    diff = np.linalg.eigvalsh(sw.covariance - gls.covariance)
    assert diff.min() > -1e-12                                                  # the unweighted estimator is never better than GLS
    marginal_only = fit_parameter_covariance(X, mode="sandwich", cov_y=_ms(np.sqrt(np.diag(cov))))
    assert not np.allclose(marginal_only.covariance, sw.covariance, rtol=1e-3)  # ignoring correlations changes the answer


# ------------------------------------------------------------------------------------------------ rank deficiency
def test_rank_deficient_fit_is_refused_honestly_without_pseudoinverse():
    xdup = np.column_stack([np.ones(6), np.ones(6)])                            # two parameters that only enter as a sum
    for mode, kw in (("sandwich", dict(cov_y=_ms(1.0))), ("residual_variance", dict(residuals=np.arange(6.0))),
                     ("weighted", dict(cov_y=_ms(1.0), objective_is_weighted=True))):
        res = fit_parameter_covariance(xdup, mode=mode, param_names=("p1", "p2"), **kw)
        assert res.covariance is None and res.std is None and not res.available
        assert res.kind == "refused" and res.status == "refused" and "not identifiable" in res.reason
        assert set(res.identifiability.weak_directions[0].components) == {"p1", "p2"}
    zero_col = np.column_stack([np.ones(6), np.zeros(6)])
    assert fit_parameter_covariance(zero_col, mode="sandwich", cov_y=_ms(1.0)).kind == "refused"


def test_ill_conditioned_fit_is_flagged_then_refused_by_the_documented_thresholds():
    def jac(eps):
        return np.vstack([np.eye(2) * np.array([1.0, eps]), np.array([[0.0, 0.0]])])
    ok = fit_parameter_covariance(jac(1e-7), mode="sandwich", cov_y=_ms(1.0, n=3))
    assert ok.status == "flagged" and ok.available                              # cond 1e7 >= 1e6
    bad = fit_parameter_covariance(jac(1e-9), mode="sandwich", cov_y=_ms(1.0, n=3))
    assert bad.status == "refused" and bad.covariance is None                   # cond 1e9 >= 1e8


# ------------------------------------------------------------------------------------------------ zero degrees of freedom
def test_exact_mode_is_A_C_At_and_equals_the_weighted_formula_for_a_positive_definite_covariance():
    j = np.array([[2.0, 1.0], [0.5, 3.0]])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    ex = fit_parameter_covariance(j, mode="exact", cov_y=_ms(None, n=2, cov=cov))
    a = np.linalg.inv(j)
    assert ex.dof == 0 and ex.kind == "exact" and ex.chi2 is None and ex.rss is None
    assert np.allclose(ex.covariance, a @ cov @ a.T, rtol=1e-12)
    wt = fit_parameter_covariance(j, mode="weighted", cov_y=_ms(None, n=2, cov=cov), objective_is_weighted=True,
                                  residuals=np.zeros(2))
    assert np.allclose(ex.covariance, wt.covariance, rtol=1e-12)
    assert wt.chi2 == 0.0 and wt.chi2_per_dof is None and any("zero degrees of freedom" in m for m in wt.messages)
    sw = fit_parameter_covariance(j, mode="sandwich", cov_y=_ms(None, n=2, cov=cov))
    assert np.allclose(sw.covariance, ex.covariance, rtol=1e-12)               # square: all three coincide


def test_exact_mode_accepts_singular_psd_covariance_and_requires_a_square_jacobian():
    j = np.array([[1.0, 0.0], [1.0, 1.0]])
    cov = np.ones((2, 2))                                                       # common shift, rank one
    res = fit_parameter_covariance(j, mode="exact", cov_y=_ms(None, n=2, cov=cov))
    a = np.linalg.inv(j)
    assert np.allclose(res.covariance, a @ cov @ a.T) and res.available
    with pytest.raises(ValueError, match="square"):
        fit_parameter_covariance(X, mode="exact", cov_y=_ms(1.0))


def test_residual_variance_is_refused_without_degrees_of_freedom():
    j = np.array([[2.0, 1.0], [0.5, 3.0]])
    res = fit_parameter_covariance(j, mode="residual_variance", residuals=np.array([0.1, -0.2]))
    assert res.covariance is None and res.kind == "refused" and "N - p" in res.reason and res.dof == 0


# ------------------------------------------------------------------------------------------------ residual variance
def test_residual_variance_formula_with_a_residual_of_prescribed_rss():
    r0 = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    proj = X @ np.linalg.solve(X.T @ X, X.T @ r0)
    r = r0 - proj                                                               # orthogonal to the model columns
    r = r * (2.0 / np.linalg.norm(r))                                           # RSS = 4 exactly
    res = fit_parameter_covariance(X, mode="residual_variance", residuals=r)
    s2 = 4.0 / 4
    assert res.rss == pytest.approx(4.0) and res.s2 == pytest.approx(s2) and res.dof == 4
    assert np.allclose(res.covariance, s2 * np.linalg.inv(X.T @ X), rtol=1e-12)
    assert res.s2_relative_uncertainty == pytest.approx(1 / np.sqrt(8))
    assert res.provenance == "residual_scaled" and not res.is_scenario and "RESIDUAL-SCALED" in res.label
    assert "not a measurement uncertainty" in res.label


def test_residual_variance_is_never_automatic_and_exact_residuals_are_not_zero_uncertainty():
    with pytest.raises(ValueError, match="residuals"):
        fit_parameter_covariance(X, mode="residual_variance")
    with pytest.raises(ValueError, match="cov_y is not accepted"):
        fit_parameter_covariance(X, mode="residual_variance", residuals=np.ones(6), cov_y=_ms(1.0))
    res = fit_parameter_covariance(X, mode="residual_variance", residuals=np.zeros(6))
    assert res.covariance is None and res.kind == "refused" and "not a statement of zero uncertainty" in res.reason


# ------------------------------------------------------------------------------------------------ unavailable versus zero
def test_unavailable_covariance_is_none_with_a_reason_and_differs_from_a_valid_zero_covariance():
    for mode in ("sandwich", "weighted", "exact"):
        with pytest.raises(ValueError, match="cov_y"):
            fit_parameter_covariance(np.eye(2) if mode == "exact" else X, mode=mode, objective_is_weighted=True)   # nothing inferred
    for kind in ("not_computed", "not_applicable", "not_available", "refused"):
        u = FitCovariance.unavailable(kind, "no defensible measurement uncertainties exist", param_names=("a",))
        assert u.covariance is None and not u.available and u.kind == kind and "NO COVARIANCE" in u.label
    with pytest.raises(ValueError):
        FitCovariance.unavailable("not_computed", "  ")
    with pytest.raises(ValueError):
        FitCovariance.unavailable("zero", "reason")
    zero = fit_parameter_covariance(X, mode="sandwich", cov_y=_ms(0.0))         # exactly known observations
    assert zero.available and zero.status == "ok" and np.all(zero.covariance == 0.0) and np.all(zero.std == 0.0)
    assert np.all(np.isnan(zero.correlation))                                    # correlation of a zero variance is undefined


def test_results_are_read_only():
    res = fit_parameter_covariance(X, mode="sandwich", cov_y=_ms(1.0))
    with pytest.raises(ValueError):
        res.covariance[0, 0] = 0.0
    with pytest.raises(Exception):
        res.status = "ok"


# ------------------------------------------------------------------------------------------------ provenance
def test_per_element_provenance_and_backward_compatibility():
    old = MeasurementSet(("a", "b"), [1.0, 2.0], np.diag([1.0, 1.0]), "", "measured", "legacy positional construction")
    assert old.kind == "measured" and old.element_kinds == ("measured", "measured") and not old.is_scenario and not old.is_mixed
    mixed = MeasurementSet.independent(("a", "b", "c"), [1, 2, 3], [1e-3, 1e-3, 5e-4], ["scenario", "measured", "scenario"],
                                       "mixed test", units="cm^-1")
    assert mixed.kind == "mixed" and mixed.is_mixed and mixed.is_scenario
    assert mixed.scenario_elements == ("a", "c") and mixed.kinds_by_name()["b"] == "measured"
    assert "MIXED" in mixed.describe() and "SCENARIO-dependent" in mixed.describe() and "measured: b" in mixed.describe()
    nonscen = MeasurementSet.independent(("a", "b"), [1, 2], 1.0, ["measured", "theory"], "no scenario")
    assert nonscen.kind == "mixed" and not nonscen.is_scenario and "SCENARIO" not in nonscen.describe()
    uniform = MeasurementSet.independent(("a", "b"), [1, 2], 1.0, ["scenario", "scenario"], "uniform")
    assert uniform.kind == "scenario" and uniform.is_scenario and not uniform.is_mixed
    with pytest.raises(ValueError):
        MeasurementSet.independent(("a", "b"), [1, 2], 1.0, ["measured"], "wrong length")
    with pytest.raises(ValueError):
        MeasurementSet.independent(("a", "b"), [1, 2], 1.0, ["measured", "guess"], "bad kind")
    with pytest.raises(ValueError):
        MeasurementSet.independent(("a", "b"), [1, 2], 1.0, "guess", "bad kind")


def test_scenario_qualification_survives_mixed_provenance_in_fit_and_inverse_map_results():
    ms = MeasurementSet.independent([f"y{i}" for i in range(6)], np.zeros(6), 0.1, ["measured"] * 5 + ["scenario"], "one assumed")
    res = fit_parameter_covariance(X, mode="sandwich", cov_y=ms)
    assert res.provenance == "mixed" and res.is_scenario and "SENSITIVITY SCENARIO" in res.label
    allmeas = fit_parameter_covariance(X, mode="sandwich", cov_y=_ms(0.1, kind="measured"))
    assert allmeas.provenance == "measured" and not allmeas.is_scenario and "SENSITIVITY SCENARIO" not in allmeas.label
    inv = inverse_map_covariance(lambda y: np.array([y.sum()]), MeasurementSet.independent(
        ("a", "b"), [1.0, 2.0], 0.1, ["measured", "scenario"], "mixed inputs"))
    assert inv.is_scenario and inv.uncertainty_kind == "mixed" and "MIXED" in inv.label and "b" in inv.label
    bare = fit_parameter_covariance(X, mode="sandwich", cov_y=np.eye(6))
    assert bare.provenance == "unspecified" and not bare.is_scenario and any("bare-array" in m for m in bare.messages)


# ------------------------------------------------------------------------------------------------ ordering contract
def test_reordered_measurement_set_permutes_everything_together():
    cov = np.array([[1.0, 0.2, 0.0], [0.2, 4.0, 0.5], [0.0, 0.5, 9.0]])
    ms = MeasurementSet.from_covariance(("c", "a", "b"), [30.0, 10.0, 20.0], cov, ["measured", "scenario", "theory"], "ordering",
                                        units=("u3", "u1", "u2"))
    order = np.argsort(ms.values)                                               # sort by value: a, b, c
    sorted_ms = ms.reordered(order)
    assert sorted_ms.names == ("a", "b", "c") and list(sorted_ms.values) == [10.0, 20.0, 30.0]
    assert sorted_ms.units == ("u1", "u2", "u3") and sorted_ms.element_kinds == ("scenario", "theory", "measured")
    assert np.array_equal(sorted_ms.cov, cov[np.ix_(order, order)])
    assert sorted_ms.cov[0, 1] == cov[1, 2] and sorted_ms.cov[0, 0] == 4.0      # a-b covariance and a variance moved together
    for bad in ([0, 1], [0, 0, 1], [0, 1, 3], [[0, 1, 2]], [0.0, 1.0, 2.0]):
        with pytest.raises(ValueError, match="permutation"):
            ms.reordered(bad)


def test_covariance_must_follow_the_observation_order_used_by_the_model():
    sig = np.array([3.0, 1.0, 2.0, 0.5])                                        # measurement order (unsorted values)
    values = np.array([40.0, 10.0, 30.0, 20.0])
    names = [f"m{i}" for i in range(4)]
    ms = MeasurementSet.independent(names, values, sig, "scenario", "unsorted measurements")
    order = np.argsort(values)                                                  # the model returns levels sorted ascending
    jac = np.column_stack([np.ones(4), np.array([0.0, 1.0, 2.0, 3.0])])        # rows follow the SORTED observations
    sorted_names = tuple(np.array(names)[order])
    good = fit_parameter_covariance(jac, mode="sandwich", cov_y=ms, measurement_order=order, observation_names=sorted_names)
    manual = fit_parameter_covariance(jac, mode="sandwich", cov_y=np.diag(sig[order] ** 2))
    wrong = fit_parameter_covariance(jac, mode="sandwich", cov_y=ms)            # unsorted covariance against sorted rows
    assert np.allclose(good.covariance, manual.covariance, rtol=1e-13)
    assert not np.allclose(good.covariance, wrong.covariance, rtol=1e-3)         # the ordering contract matters
    assert good.observation_names == sorted_names and not any("not verified" in m for m in good.messages)
    assert any("not verified" in m for m in wrong.messages)
    with pytest.raises(ValueError, match="ordering contract"):
        fit_parameter_covariance(jac, mode="sandwich", cov_y=ms, observation_names=sorted_names)     # names not permuted
    bare = fit_parameter_covariance(jac, mode="sandwich", cov_y=np.diag(sig ** 2), measurement_order=order)
    assert np.allclose(bare.covariance, manual.covariance, rtol=1e-13)
    with pytest.raises(ValueError):
        fit_parameter_covariance(jac, mode="sandwich", cov_y=np.eye(3))        # dimension mismatch
    with pytest.raises(ValueError, match="permutation"):
        fit_parameter_covariance(jac, mode="sandwich", cov_y=ms, measurement_order=[0, 1, 2, 2])


# ------------------------------------------------------------------------------------------------ input validation, physical Jacobian
def test_input_validation():
    with pytest.raises(ValueError, match="mode"):
        fit_parameter_covariance(X, mode="gls", cov_y=_ms(1.0))
    with pytest.raises(ValueError, match="non-finite"):
        fit_parameter_covariance(np.where(X > 3, np.nan, X), mode="sandwich", cov_y=_ms(1.0))
    with pytest.raises(ValueError, match="param_names"):
        fit_parameter_covariance(X, mode="sandwich", cov_y=_ms(1.0), param_names=("only",))
    with pytest.raises(ValueError, match="residuals"):
        fit_parameter_covariance(X, mode="weighted", cov_y=_ms(1.0), objective_is_weighted=True, residuals=np.ones(5))
    with pytest.raises(ValueError, match="2-D"):
        fit_parameter_covariance(np.ones(3), mode="sandwich", cov_y=_ms(1.0, n=3))
    with pytest.raises(CovarianceError):
        fit_parameter_covariance(X, mode="sandwich", cov_y=-np.eye(6))


def test_works_on_the_jacobian_of_the_real_zeeman_model_without_touching_the_fit_routine():
    from src.solvers import energy_vs_field
    from src.validation import zeta_from_fine_structure
    zeta = zeta_from_fine_structure(17.196, l=1)
    b = np.linspace(0.5, 30, 12)
    jr = numerical_jacobian(lambda z: energy_vs_field(1, b, z[0]).ravel(), np.array([zeta]), steps=[zeta * 1e-4])
    assert jr.stable
    sig = 1e-6                                                                   # an implementation scenario, not a measurement
    res = fit_parameter_covariance(jr.jac, mode="sandwich", cov_y=MeasurementSet.independent(
        [f"e{i}" for i in range(72)], np.zeros(72), sig, "scenario", "prescribed 1 micro-eV per level (implementation scenario)"),
        param_names=("zeta_eV",), param_values=[zeta])
    j = jr.jac[:, 0]
    assert res.std[0] == pytest.approx(sig / np.linalg.norm(j), rel=1e-12)       # iid: sigma / |J|
    assert res.is_scenario and res.dof == 71
