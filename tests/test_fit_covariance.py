"""Stage 8B.1: covariance infrastructure. Deterministic, analytically solvable tests (no random coverage thresholds).

These are MATHEMATICAL VERIFICATION of the implementation. None of them is an experimental validation.
"""
import numpy as np
import pytest

from src.fit_covariance import (CovarianceError, JacobianError, MeasurementSet, identifiability, inverse_map_covariance,
                                numerical_jacobian, uncertainty_sensitivity, validate_covariance)
from src.two_electron import CA_LEVELS_CM1, calcium_fit_sensitivity, calcium_four_parameter_fit

pytestmark = pytest.mark.implementation

KEYS = ("3P0", "3P1", "3P2", "1P1")
OUT = ["zeta_S", "zeta_A", "Delta", "E_t", "eps2"]


def _ms(vals, sigma, kind="measured", names=None):
    names = names or [f"y{i}" for i in range(len(vals))]
    return MeasurementSet.independent(names, vals, sigma, kind, "unit test")


# ------------------------------------------------------------------------------------------ covariance validation
def test_validate_accepts_psd_including_singular():
    ones = np.ones((3, 3))                                         # rank 1: perfectly correlated inputs, valid for propagation
    c = validate_covariance(ones)
    assert c.is_psd and not c.is_pd and c.rank == 1
    assert validate_covariance(np.diag([1.0, 4.0])).is_pd
    assert validate_covariance(np.zeros((2, 2))).rank == 0         # zero matrix is positive semidefinite


def test_validate_rejects_non_symmetric_negative_eigenvalue_nonfinite_and_shape():
    with pytest.raises(CovarianceError, match="not symmetric"):
        validate_covariance([[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(CovarianceError, match="not positive semidefinite"):
        validate_covariance([[1.0, 2.0], [2.0, 1.0]])             # eigenvalues 3 and -1
    with pytest.raises(CovarianceError, match="non-finite"):
        validate_covariance([[np.nan, 0.0], [0.0, 1.0]])
    with pytest.raises(CovarianceError, match="square"):
        validate_covariance(np.ones((2, 3)))


def test_tolerances_are_scale_aware():
    big = 1e12 * np.array([[1.0, 0.5], [0.5, 1.0]])
    big[0, 1] += 1e-3                                              # asymmetry 1e-15 of the scale: fine
    assert validate_covariance(big).is_psd
    tiny = 1e-12 * np.array([[1.0, 2.0], [2.0, 1.0]])             # same structure at a tiny scale is still rejected
    with pytest.raises(CovarianceError):
        validate_covariance(tiny)


# ------------------------------------------------------------------------------------------------ MeasurementSet
def test_measurement_set_constructors_and_provenance():
    a = _ms([1.0, 2.0], [0.1, 0.2], kind="theory")
    assert np.allclose(a.cov, np.diag([0.01, 0.04])) and np.allclose(a.sigma, [0.1, 0.2])
    assert a.kind == "theory" and not a.is_scenario and "theory covariance" in a.describe()
    b = MeasurementSet.from_correlation(["u", "v"], [0.0, 0.0], [1.0, 2.0], [[1.0, 0.5], [0.5, 1.0]], "scenario", "assumed", units="cm^-1")
    assert b.cov[0, 1] == pytest.approx(1.0) and b.units == ("cm^-1", "cm^-1")
    assert b.is_scenario and "SCENARIO" in b.describe() and "not an experimentally established" in b.describe()
    c = MeasurementSet.from_covariance(["u", "v"], [0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]], "recommended", "src")      # singular is accepted
    assert c.check().rank == 1


def test_measurement_set_rejects_bad_input():
    with pytest.raises(ValueError, match="kind"):
        MeasurementSet.independent(["a"], [1.0], 0.1, "assumed", "src")
    with pytest.raises(ValueError, match="source"):
        MeasurementSet.independent(["a"], [1.0], 0.1, "measured", "  ")
    with pytest.raises(ValueError, match="unique"):
        MeasurementSet.independent(["a", "a"], [1.0, 2.0], 0.1, "measured", "s")
    with pytest.raises(ValueError, match="non-finite"):
        MeasurementSet.independent(["a"], [np.inf], 0.1, "measured", "s")
    with pytest.raises(ValueError):
        MeasurementSet.independent(["a"], [1.0], -0.1, "measured", "s")
    with pytest.raises(CovarianceError):
        MeasurementSet.from_correlation(["a", "b"], [0.0, 0.0], 1.0, [[1.0, 1.5], [1.5, 1.0]], "measured", "s")
    with pytest.raises(CovarianceError):
        MeasurementSet.from_covariance(["a", "b"], [0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], "measured", "s")


def test_arrays_are_read_only_and_cannot_be_reenabled():
    vals, cov = np.array([1.0, 2.0]), np.diag([1.0, 1.0])
    m = MeasurementSet.from_covariance(["a", "b"], vals, cov, "measured", "s")
    vals[0], cov[0, 0] = 99.0, 99.0                                # caller's arrays do not leak into the set
    assert m.values[0] == 1.0 and m.cov[0, 0] == 1.0
    for arr in (m.values, m.cov, m.sigma, m.correlation()):
        with pytest.raises(ValueError):
            arr[0] = 5.0
        with pytest.raises(ValueError):
            arr.setflags(write=True)
    with pytest.raises(Exception):
        m.kind = "scenario"                                        # frozen dataclass


def test_no_correlation_is_inferred_and_zero_variance_gives_nan():
    m = _ms([1.0, 2.0, 3.0], [0.1, 0.1, 0.0])
    r = m.correlation()
    assert r[0, 1] == 0.0 and np.isnan(r[2, 2]) and np.isnan(r[0, 2])


def test_positive_definite_requirement_for_weighted_fitting():
    ok = _ms([0.0, 0.0], [1.0, 2.0])
    ok.require_positive_definite()
    lo = ok.whitening_factor()
    assert np.allclose(lo @ lo.T, ok.cov)
    singular = MeasurementSet.from_covariance(["a", "b"], [0.0, 0.0], np.ones((2, 2)), "scenario", "s")
    with pytest.raises(CovarianceError, match="singular.*no regularisation"):
        singular.require_positive_definite()
    with pytest.raises(CovarianceError):
        singular.whitening_factor()


# --------------------------------------------------------------------------------- inverse-map propagation (linear)
M = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])


def lin(y):
    return M @ y


def test_independent_and_correlated_linear_propagation_exact():
    sig = np.array([0.3, 0.2, 0.5])
    ind = MeasurementSet.independent(["a", "b", "c"], [1.0, 2.0, 3.0], sig, "measured", "t")
    r = inverse_map_covariance(lin, ind)
    assert r.status == "ok" and r.jacobian_source.startswith("numerical")
    assert np.allclose(r.covariance, M @ np.diag(sig ** 2) @ M.T, rtol=1e-7, atol=1e-12)
    corr = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    cor = MeasurementSet.from_correlation(["a", "b", "c"], [1.0, 2.0, 3.0], sig, corr, "measured", "t")
    rc = inverse_map_covariance(lin, cor, jacobian=M)               # analytical Jacobian
    assert rc.jacobian_source.startswith("analytical")
    assert np.allclose(rc.covariance, M @ cor.cov @ M.T, rtol=1e-14, atol=1e-14)
    assert not np.allclose(rc.covariance, r.covariance)             # ignoring the correlation changes the answer
    rn = inverse_map_covariance(lin, cor)
    assert np.allclose(rn.covariance, rc.covariance, rtol=1e-6, atol=1e-10)


def test_positive_semidefinite_singular_input_is_propagated():
    full = MeasurementSet.from_covariance(["a", "b", "c"], [1.0, 2.0, 3.0], 0.04 * np.ones((3, 3)), "scenario", "common shift")
    r = inverse_map_covariance(lin, full, outputs_are_parameters=False)
    assert np.allclose(r.covariance, 0.04 * (M @ np.ones((3, 3)) @ M.T), rtol=1e-6, atol=1e-8)      # numerical Jacobian: ~1e-8 accuracy
    assert validate_covariance(r.covariance).is_psd


def test_nonlinear_map_matches_analytic_jacobian():
    f = lambda y: np.array([np.exp(y[0]) * y[1], np.sin(y[1])])
    y0 = np.array([0.3, 1.2])
    jac = np.array([[np.exp(0.3) * 1.2, np.exp(0.3)], [0.0, np.cos(1.2)]])
    r = inverse_map_covariance(f, _ms(y0, [0.01, 0.02]))
    assert np.allclose(r.jacobian, jac, rtol=1e-6)
    assert np.allclose(r.covariance, jac @ np.diag([1e-4, 4e-4]) @ jac.T, rtol=1e-6)


def test_provenance_and_scenario_labelling_are_carried():
    sc = inverse_map_covariance(lin, _ms([1.0, 2.0, 3.0], 0.1, kind="scenario"))
    me = inverse_map_covariance(lin, _ms([1.0, 2.0, 3.0], 0.1, kind="measured"))
    assert sc.is_scenario and sc.uncertainty_kind == "scenario" and "SENSITIVITY SCENARIO" in sc.label and "not an experimental" in sc.label
    assert not me.is_scenario and "measured" in me.label and "SCENARIO" not in me.label
    assert "excludes model error" in me.label and "goodness-of-fit" in me.label
    assert sc.input_names == ("y0", "y1", "y2")


# ---------------------------------------------------------------------------------- identifiability and scaling
def test_rank_deficient_jacobian_is_refused_with_weak_direction():
    f = lambda y: np.array([y[0] + y[1], 2.0 * (y[0] + y[1])])      # both outputs depend on y0 + y1: rank 1
    r = inverse_map_covariance(f, _ms([1.0, 2.0], 0.1))
    assert r.status == "refused" and r.covariance is None and r.std is None and r.correlation is None
    assert r.identifiability.rank == 1 and np.isinf(r.identifiability.condition_number)
    w = r.identifiability.weak_directions[0]
    assert w.singular_value == pytest.approx(0.0, abs=1e-8) and "not identifiable" in r.messages[0]
    # the same Jacobian for outputs that are allowed to be dependent is flagged, and the singular covariance is returned
    d = inverse_map_covariance(f, _ms([1.0, 2.0], 0.1), outputs_are_parameters=False)
    assert d.status == "flagged" and d.covariance is not None and d.identifiability.rank == 1
    assert np.linalg.matrix_rank(d.covariance, tol=1e-12) == 1


def test_weak_direction_names_the_parameter_combination():
    a = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9]])                   # nearly parallel rows: combination p1 - p2 barely constrained
    idn = identifiability(a, orientation="inverse", param_names=["p1", "p2"], scales=[1.0, 1.0])
    assert idn.status == "refused" and idn.condition_number > 1e8
    top = idn.weak_directions[0]
    assert set(top.components) == {"p1", "p2"} and abs(abs(top.components["p1"]) - 2 ** -0.5) < 1e-6
    assert top.components["p1"] * top.components["p2"] < 0           # the antisymmetric combination


def test_poorly_conditioned_is_flagged_then_refused_by_thresholds():
    def cond_matrix(kappa):
        return np.diag([1.0, 1.0 / kappa])
    flagged = identifiability(cond_matrix(1e7))
    refused = identifiability(cond_matrix(1e9))
    ok = identifiability(cond_matrix(1e3))
    assert (ok.status, flagged.status, refused.status) == ("ok", "flagged", "refused")
    assert flagged.condition_number == pytest.approx(1e7, rel=1e-9)
    assert identifiability(cond_matrix(1e9), flag_cond=1e12, refuse_cond=1e14).status == "ok"     # thresholds are conventions
    assert refused.refuse_cond == 1e8 and refused.flag_cond == 1e6


def test_parameter_scaling_changes_the_condition_number_and_is_reported():
    a = np.diag([1.0, 1e6])                                          # parameters in very different units
    raw = identifiability(a, orientation="forward")
    scaled = identifiability(a, orientation="forward", scales=[1.0, 1e-6])
    assert raw.condition_number == pytest.approx(1e6, rel=1e-9) and raw.status == "flagged"
    assert scaled.condition_number == pytest.approx(1.0, rel=1e-9) and scaled.status == "ok"
    assert "S_p = 1" in raw.scaling and "given explicitly" in scaled.scaling and "NOT whitened" in raw.scaling
    with pytest.raises(ValueError):
        identifiability(a, scales=[1.0, -1.0])


def test_forward_whitening_matches_the_analytic_singular_value():
    j = np.array([[1.0], [1.0]])
    idn = identifiability(j, cov_y=np.diag([1.0, 4.0]), scales=[2.0])   # L^-1 J S_p = [2, 1]
    assert idn.singular_values[0] == pytest.approx(np.sqrt(5.0), rel=1e-12)
    assert "whitened" in idn.scaling and "NOT" not in idn.scaling
    with pytest.raises(CovarianceError, match="positive-definite"):
        identifiability(j, cov_y=np.ones((2, 2)))                        # singular covariance is not regularised


def test_underdetermined_forward_problem_is_refused_and_nonfinite_jacobian_refused():
    assert identifiability(np.array([[1.0, 2.0]])).status == "refused"     # 1 observation, 2 parameters
    assert identifiability(np.array([[np.nan]])).status == "refused"
    assert identifiability(np.zeros((2, 2))).status == "refused"


def test_correlation_flag_only_when_parameters_should_be_independent():
    r = np.array([[1.0, 0.995], [0.995, 1.0]])
    a = identifiability(np.eye(2), correlation=r)
    b = identifiability(np.eye(2), correlation=r, expect_independent_params=False)
    assert a.status == "flagged" and a.max_abs_correlation == pytest.approx(0.995)
    assert b.status == "ok"


# ------------------------------------------------------------------------------------ finite-difference stability
def test_step_stability_check_passes_for_smooth_maps_and_refuses_kinks():
    smooth = numerical_jacobian(lambda y: np.array([np.exp(y[0]) * y[1]]), [0.3, 1.2])
    assert smooth.stable and smooth.spread < 1e-7
    kink = numerical_jacobian(lambda y: np.array([abs(y[0] - 1e-8)]), [0.0], steps=[1e-8])   # kink lies inside the step
    assert not kink.stable and kink.spread > 1e-3
    r = inverse_map_covariance(lambda y: np.array([abs(y[0] - 1e-8)]), _ms([0.0], 0.1), steps=[1e-8])
    assert r.status == "refused" and r.covariance is None and "not step-stable" in " ".join(r.messages)


def test_result_does_not_depend_on_the_base_step_within_tolerance():
    f = lambda y: np.array([y[0] ** 3 / (1.0 + y[1] ** 2), np.log(y[0] + 2.0)])
    a1 = numerical_jacobian(f, [0.7, 0.4], rel_step=1e-6).jac
    a2 = numerical_jacobian(f, [0.7, 0.4], rel_step=1e-8).jac
    assert np.allclose(a1, a2, rtol=1e-6, atol=1e-8)


def test_map_undefined_near_the_domain_boundary_raises_jacobian_error():
    def f(y):
        if y[0] < 0:
            raise ValueError("outside the domain")
        return np.array([np.sqrt(y[0])])
    with pytest.raises(JacobianError, match="boundary of its domain"):
        numerical_jacobian(f, [1e-12], steps=[1e-9])
    with pytest.raises(JacobianError):
        numerical_jacobian(lambda y: np.array([np.nan]), [1.0])


def test_supplied_jacobian_shape_and_finiteness_are_validated():
    with pytest.raises(JacobianError):
        inverse_map_covariance(lin, _ms([1.0, 2.0, 3.0], 0.1), jacobian=np.ones((3, 3)))
    with pytest.raises(JacobianError):
        inverse_map_covariance(lin, _ms([1.0, 2.0, 3.0], 0.1), jacobian=np.full((2, 3), np.nan))


# ------------------------------------------------------------------------------------------ calcium regression
def _ca_map(names):
    return lambda v: np.array([calcium_four_parameter_fit(*v)[n] for n in names])


def _ca_set(sigma=0.001, kind="scenario"):
    src = "assumed independent scenario of 0.001 cm^-1 per level at about the tabulated precision; NIST publishes no uncertainties"
    return MeasurementSet.independent(KEYS, [CA_LEVELS_CM1[k] for k in KEYS], sigma, kind, src, units="cm^-1")


def test_calcium_matches_calcium_fit_sensitivity_for_all_five_outputs():
    ref = calcium_fit_sensitivity(CA_LEVELS_CM1, 0.001)
    r = inverse_map_covariance(_ca_map(OUT), _ca_set(), output_names=OUT, output_units=["cm^-1"] * 4 + [""], outputs_are_parameters=False)
    assert r.status == "flagged" and r.identifiability.rank == 4                # eps2 is a function of the four levels
    assert np.allclose(r.values, ref.value.values, rtol=1e-12)
    assert np.allclose(r.std, ref.scenario_std.values, rtol=1e-6)
    assert np.allclose(r.correlation, ref.attrs["correlation"].values, atol=1e-6)
    assert np.allclose(np.sqrt(np.diag(r.covariance)), r.std)
    assert r.is_scenario and "SENSITIVITY SCENARIO" in r.label and "NIST" in _ca_set().source


def test_calcium_four_parameters_alone_are_identifiable_and_ok():
    names = OUT[:4]
    r = inverse_map_covariance(_ca_map(names), _ca_set(), output_names=names)
    assert r.status == "ok" and r.identifiability.rank == 4
    ref = calcium_fit_sensitivity(CA_LEVELS_CM1, 0.001).loc[names]
    assert np.allclose(r.std, ref.scenario_std.values, rtol=1e-6)
    assert r.jacobian_step_spread < 1e-5 and "numerical" in r.jacobian_source
    assert "S_p = |p_hat|" in r.identifiability.scaling


def test_calcium_eps2_analytic_jacobian_and_sigma():
    l0, l1, l2, lp = (CA_LEVELS_CM1[k] for k in KEYS)
    a, d = (l2 + 2 * l0) / 3, lp - l1
    g = np.array([(2 / 3) / d, -(lp - a) / d ** 2, (1 / 3) / d, -(a - l1) / d ** 2])
    r = inverse_map_covariance(_ca_map(["eps2"]), _ca_set(), output_names=["eps2"], outputs_are_parameters=False)
    assert np.allclose(r.jacobian[0], g, rtol=1e-6)
    assert r.std[0] == pytest.approx(1e-3 * np.sqrt((g ** 2).sum()), rel=1e-6) and r.std[0] == pytest.approx(1.477e-7, rel=1e-3)
    ra = inverse_map_covariance(_ca_map(["eps2"]), _ca_set(), output_names=["eps2"], jacobian=g[None, :], outputs_are_parameters=False)
    assert ra.std[0] == pytest.approx(r.std[0], rel=1e-6)


def test_calcium_per_level_sigma_dict_equivalent_to_independent_vector():
    sig = {"3P0": 0.002, "3P1": 0.0005, "3P2": 0.001, "1P1": 0.004}
    ref = calcium_fit_sensitivity(CA_LEVELS_CM1, sig)
    ms = MeasurementSet.independent(KEYS, [CA_LEVELS_CM1[k] for k in KEYS], [sig[k] for k in KEYS], "scenario", "assumed scenario")
    r = inverse_map_covariance(_ca_map(OUT), ms, output_names=OUT, outputs_are_parameters=False)
    assert np.allclose(r.std, ref.scenario_std.values, rtol=1e-6)


def test_calcium_correlated_scenario_against_analytic_expressions():
    s, rho = 0.001, 0.5
    corr = np.full((4, 4), rho) + (1 - rho) * np.eye(4)
    ms = MeasurementSet.from_correlation(KEYS, [CA_LEVELS_CM1[k] for k in KEYS], s, corr, "scenario", "assumed correlated scenario")
    r = inverse_map_covariance(_ca_map(OUT), ms, output_names=OUT, outputs_are_parameters=False)
    l0, l1, l2, lp = (CA_LEVELS_CM1[k] for k in KEYS)
    a, d = (l2 + 2 * l0) / 3, lp - l1
    g = np.array([(2 / 3) / d, -(lp - a) / d ** 2, (1 / 3) / d, -(a - l1) / d ** 2])
    assert r.covariance[4, 4] == pytest.approx(g @ ms.cov @ g, rel=1e-6)
    assert r.covariance[0, 0] == pytest.approx(2 * s ** 2 * (1 - rho) / 9, rel=1e-6)        # zeta_S = (E3P2 - E3P0)/3
    ind = inverse_map_covariance(_ca_map(OUT), _ca_set(), output_names=OUT, outputs_are_parameters=False)
    assert not np.allclose(r.covariance, ind.covariance, rtol=1e-3)


def test_calcium_common_shift_of_all_levels_is_exactly_known():
    """All four levels shifted together (rank-1 PSD covariance): intervals do not change, only E_t moves."""
    s = 0.001
    ms = MeasurementSet.from_covariance(KEYS, [CA_LEVELS_CM1[k] for k in KEYS], s ** 2 * np.ones((4, 4)), "scenario", "assumed common shift")
    r = inverse_map_covariance(_ca_map(OUT), ms, output_names=OUT, outputs_are_parameters=False)
    var = np.diag(r.covariance)
    assert var[3] == pytest.approx(s ** 2, rel=1e-6)                                         # E_t shifts by the common amount
    for k in (0, 1, 2, 4):                                                                  # zeta_S, zeta_A, Delta, eps2 unchanged
        assert var[k] < 1e-12 * s ** 2 + 1e-18


def test_calcium_boundary_is_reported_as_a_domain_error_not_a_number():
    """3P1 placed 1e-9 cm^-1 below its interval-rule position: any step crosses the V^2 >= 0 boundary."""
    l0, l2, lp = CA_LEVELS_CM1["3P0"], CA_LEVELS_CM1["3P2"], CA_LEVELS_CM1["1P1"]
    a = (l2 + 2 * l0) / 3
    ms = MeasurementSet.independent(KEYS, [l0, a - 1e-9, l2, lp], 0.001, "scenario", "boundary test")
    with pytest.raises(JacobianError, match="boundary of its domain"):
        inverse_map_covariance(_ca_map(OUT), ms, output_names=OUT, outputs_are_parameters=False)


def test_existing_calcium_interface_is_unchanged():
    df = calcium_fit_sensitivity(CA_LEVELS_CM1, 0.001)
    assert list(df.columns) == ["value", "scenario_std", "std_err"] and list(df.index) == OUT


# ------------------------------------------------------------------ review corrections: zero covariance, two diagnostics
def test_zero_input_covariance_gives_zero_output_covariance_not_a_refusal():
    ms = MeasurementSet.from_covariance(["y"], [2.0], [[0.0]], "measured", "exact input")
    r = inverse_map_covariance(lambda y: 3.0 * y, ms, output_names=["p"])
    assert r.status == "ok" and r.covariance is not None
    assert r.covariance[0, 0] == 0.0 and r.std[0] == 0.0
    assert r.identifiability.rank == 1 and r.identifiability.status == "ok"
    assert r.sensitivity.is_zero and r.sensitivity.covariance_rank == 0
    ind = inverse_map_covariance(lambda y: 3.0 * y, _ms([2.0], [0.0]))          # same via the independent constructor
    assert ind.status == "ok" and ind.covariance[0, 0] == 0.0


def test_structural_diagnostic_is_independent_of_the_supplied_covariance():
    f = lambda y: np.array([y[0] + y[1], y[0] - 2 * y[1]])
    ranks, conds = set(), set()
    for sig in ([1.0, 1.0], [0.0, 0.0], [1e-12, 5.0], [0.0, 1.0]):
        r = inverse_map_covariance(f, _ms([1.0, 2.0], sig))
        assert r.status in ("ok", "flagged") and r.covariance is not None
        ranks.add(r.identifiability.rank)
        conds.add(round(r.identifiability.condition_number, 9))
    assert ranks == {2} and len(conds) == 1
    assert "independent of the input covariance" in r.identifiability.scaling
    with pytest.raises(ValueError):
        identifiability(np.eye(2), orientation="inverse", cov_y=np.eye(2))      # the covariance is not a structural input


def test_zero_variance_direction_is_not_a_structural_failure_but_rank_deficiency_still_refuses():
    ms = _ms([1.0, 1.0], [1.0, 0.0])                                             # second input exact
    r = inverse_map_covariance(lambda y: np.array([y[0] + y[1], y[0] - y[1]]), ms)
    assert r.status != "refused" and r.identifiability.rank == 2
    assert r.sensitivity.covariance_rank == 1                                    # only one propagated direction
    dep = inverse_map_covariance(lambda y: np.array([y[0], 2.0 * y[0]]), _ms([1.0], [0.0]))
    assert dep.status == "refused" and dep.covariance is None and dep.identifiability.rank == 1
    dep2 = inverse_map_covariance(lambda y: np.array([y[0], 2.0 * y[0]]), _ms([1.0], [1.0]))
    assert dep2.status == "refused"                                              # unchanged by the uncertainty value


def test_correlated_inputs_propagate_as_A_C_At_and_descriptions_state_what_is_used():
    cov = np.array([[4.0, 1.5], [1.5, 1.0]])
    a = np.array([[1.0, 2.0], [3.0, -1.0]])
    ms = MeasurementSet.from_covariance(["y0", "y1"], [1.0, 1.0], cov, "measured", "correlated test")
    r = inverse_map_covariance(lambda y: a @ y, ms)
    assert np.allclose(r.covariance, a @ cov @ a.T, rtol=1e-6, atol=1e-8)
    sens = r.sensitivity
    assert "NOT included" in sens.marginal_description and "marginal" in sens.marginal_description
    assert "full input covariance" in sens.covariance_description
    scale = np.where(np.abs(r.values) > 0, np.abs(r.values), 1.0)
    # squared covariance-aware singular values = eigenvalues of the scaled output covariance
    scaled_cov = r.covariance / np.outer(scale, scale)
    assert np.allclose(np.sort(sens.covariance_singular_values ** 2), np.sort(np.linalg.eigvalsh(scaled_cov)), rtol=1e-6)
    # the marginal diagnostic differs from the covariance-aware one when inputs are correlated
    assert not np.allclose(np.sort(sens.marginal_singular_values), np.sort(sens.covariance_singular_values))
    uncorr = uncertainty_sensitivity(a, np.diag(np.diag(cov)))
    assert np.allclose(np.sort(uncorr.marginal_singular_values), np.sort(uncorr.covariance_singular_values))


def test_singular_psd_covariance_sensitivity_and_propagation():
    cov = np.ones((3, 3))                                                        # rank 1
    a = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    sens = uncertainty_sensitivity(a, cov)
    assert sens.covariance_rank == 1 and not sens.is_zero
    assert np.isinf(sens.covariance_cond)
    ms = MeasurementSet.from_covariance(["a", "b", "c"], [1.0, 2.0, 3.0], cov, "scenario", "common shift")
    r = inverse_map_covariance(lambda y: np.array([y.sum(), y @ np.array([1.0, 2.0, 3.0])]), ms)
    assert r.status != "refused" and np.allclose(r.covariance, a @ cov @ a.T, rtol=1e-6, atol=1e-8)
    with pytest.raises(CovarianceError):
        uncertainty_sensitivity(a, np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]))


def test_sensitivity_reports_left_null_directions_when_outputs_exceed_inputs():
    a = np.array([[1.0], [2.0], [3.0]])
    res = uncertainty_sensitivity(a, np.array([[1.0]]), param_names=["a", "b", "c"], n_weak=3)
    assert res.covariance_rank == 1 and not res.is_zero and np.isinf(res.covariance_cond)
    assert res.covariance_singular_values.shape == (1,)
    assert len(res.weak_directions) == 2                                        # exactly the two null directions
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    vecs = []
    for w in res.weak_directions:
        assert w.singular_value == 0.0 and w.ratio_to_largest == 0.0
        assert set(w.components) <= {"a", "b", "c"}
        v = np.array([w.components.get(k, 0.0) for k in "abc"])
        assert np.all(np.isfinite(v))
        vecs.append(v)
    # components are reported for the top three entries here (m = 3), so each vector is complete: unit norm, orthogonal to range(A)
    full = [np.array([w.components[k] for k in "abc"]) for w in res.weak_directions]
    for v in full:
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
        assert abs(v @ axis) < 1e-12
    assert abs(full[0] @ full[1]) < 1e-12
    assert len(uncertainty_sensitivity(a, np.array([[1.0]]), n_weak=1).weak_directions) == 1     # documented n_weak limit


def test_sensitivity_square_and_wide_cases_unchanged_and_validation():
    sq = uncertainty_sensitivity(np.diag([1.0, 1e-9]), np.eye(2))                # m = n, one weak direction
    assert sq.covariance_rank == 2 and len(sq.weak_directions) == 1 and sq.weak_directions[0].ratio_to_largest == pytest.approx(1e-9)
    wide = uncertainty_sensitivity(np.array([[1.0, 1.0, 1.0]]), np.eye(3))       # m < n, full rank 1
    assert wide.covariance_rank == 1 and wide.weak_directions == ()
    with pytest.raises(ValueError, match="non-finite"):
        uncertainty_sensitivity(np.array([[np.nan]]), np.eye(1))
    with pytest.raises(ValueError, match="param_names"):
        uncertainty_sensitivity(np.eye(2), np.eye(2), param_names=["only_one"])
    with pytest.raises(ValueError):
        uncertainty_sensitivity(np.ones(3), np.eye(3))
