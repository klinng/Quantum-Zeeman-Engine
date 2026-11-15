r"""Core covariance infrastructure (Stage 8B.1): measurement covariance, inverse-map propagation, identifiability.

WHAT IS IMPLEMENTED
-------------------
* ``MeasurementSet``: named values with units, a validated covariance matrix and an explicit provenance
  (``measured``, ``theory``, ``recommended`` or ``scenario``). The arrays are read-only.
* ``validate_covariance``: finite values, symmetry and positive-semidefiniteness with scale-aware tolerances.
* ``numerical_jacobian``: central differences at three step sizes with a step-stability check.
* ``inverse_map_covariance``: C_p = A C_y A^T for an explicit parameter map p = p(y), A = dp/dy.
* ``identifiability``: SVD diagnostics of a scaled Jacobian (rank, condition number, weak combinations, status).

WHAT IS NOT IMPLEMENTED (deliberately)
--------------------------------------
Weighted-fit covariance (J^T C_y^-1 J)^-1, residual-variance covariance s^2 (J^T J)^-1, a general Monte Carlo engine,
helium numerical covariance, and any change to ``fit_parameters`` or ``fit_to_levels``. ``MeasurementSet.require_positive_definite``
and ``whitening_factor`` exist so that a later weighted fit can refuse singular covariances; no weighted fit uses them yet.

WHAT THE RESULT MEANS
---------------------
``inverse_map_covariance`` returns the covariance of the outputs implied by the stated input covariance, to FIRST ORDER
(local linearisation of the map at the given point). It is a conditional statement: "if the inputs had this covariance and
the model map is right". It is not a goodness-of-fit, it does not test the model, it does not estimate noise, and it does not
include model error (omitted interactions, missing perturbers) or numerical error of the map itself.
When the input provenance is ``scenario`` the output covariance is a SENSITIVITY SCENARIO: the input covariance was assumed,
not established by a measurement, so the result is not an experimental uncertainty.

The default numerical Jacobian (central differences, step 1e-8 max(|y|,1)) is accurate to roughly 1e-8 of the column scale, which is
a trade-off between truncation and round-off; the step-stability check refuses a Jacobian whose three step sizes disagree by more
than 1e-5. Supply an analytical Jacobian when exact derivatives are available.

Linear propagation is unreliable near a boundary of the map's domain or where the map is strongly curved on the scale of the
input uncertainty (for calcium, V^2 = (a - E3P1)(E1P1 - a) >= 0 sets such a boundary). A later Monte Carlo cross-check must
separate expected domain rejection from numerical failure and from programming errors, report the rejected samples and the
acceptance fraction, and label statistics of accepted samples as conditional on acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import cholesky, solve_triangular

PROVENANCE = ("measured", "theory", "recommended", "scenario")
_EPS = np.finfo(float).eps


class CovarianceError(ValueError):
    """A covariance matrix is malformed, not positive semidefinite, or singular where positive definiteness is required."""


class JacobianError(ValueError):
    """The Jacobian could not be formed reliably (non-finite values, map undefined at a perturbed point, unstable step)."""


# ---------------------------------------------------------------------------------------------------------- helpers
def _frozen(a) -> np.ndarray:
    """Read-only copy that cannot be made writable again: a view of a read-only owner array."""
    owner = np.array(a, dtype=float, copy=True)
    owner.setflags(write=False)
    return owner.view()


@dataclass(frozen=True)
class CovarianceCheck:
    """Outcome of ``validate_covariance`` (all numbers are scale-aware)."""
    n: int
    symmetric_error: float       # max |C - C^T| / max |C|
    lambda_min: float
    lambda_max: float
    rank: int                    # eigenvalues above rank_rtol * lambda_max
    is_psd: bool
    is_pd: bool


def validate_covariance(cov, *, sym_rtol: float = 1e-10, psd_rtol: float = 1e-10, rank_rtol: float = 1e-12,
                        name: str = "covariance") -> CovarianceCheck:
    """Check finiteness, symmetry and positive-semidefiniteness of a covariance matrix; raise ``CovarianceError`` if invalid.

    Tolerances are relative to the largest entry / eigenvalue: symmetric if max|C - C^T| <= sym_rtol max|C|; positive
    semidefinite if lambda_min >= -psd_rtol lambda_max. A matrix with zero eigenvalues IS accepted (it is valid for
    propagation, e.g. fully correlated inputs); ``is_pd`` reports whether it is also positive definite
    (lambda_min > rank_rtol lambda_max), which a weighted fit would require."""
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise CovarianceError(f"{name} must be a square matrix, got shape {c.shape}")
    if not np.all(np.isfinite(c)):
        raise CovarianceError(f"{name} contains non-finite values")
    scale = float(np.max(np.abs(c))) if c.size else 0.0
    sym_err = float(np.max(np.abs(c - c.T)) / scale) if scale > 0 else 0.0
    if sym_err > sym_rtol:
        raise CovarianceError(f"{name} is not symmetric: max|C - C^T|/max|C| = {sym_err:.3e} > {sym_rtol:.1e}")
    w = np.linalg.eigvalsh(0.5 * (c + c.T)) if c.size else np.array([0.0])
    lam_max, lam_min = float(w[-1]), float(w[0])
    if lam_min < -psd_rtol * max(lam_max, 0.0):
        raise CovarianceError(f"{name} is not positive semidefinite: smallest eigenvalue {lam_min:.3e} "
                              f"(largest {lam_max:.3e}), allowed down to -{psd_rtol:.1e} x largest")
    if np.any(np.diag(c) < 0):
        raise CovarianceError(f"{name} has a negative variance on the diagonal")
    rank = int(np.sum(w > rank_rtol * lam_max)) if lam_max > 0 else 0
    return CovarianceCheck(n=c.shape[0], symmetric_error=sym_err, lambda_min=lam_min, lambda_max=lam_max, rank=rank,
                           is_psd=True, is_pd=bool(rank == c.shape[0]))


# ------------------------------------------------------------------------------------------------ MeasurementSet
@dataclass(frozen=True)
class MeasurementSet:
    """Named measured (or assumed) values with a covariance matrix and an explicit provenance.

    ``kind`` is one of ``measured``, ``theory``, ``recommended``, ``scenario``. A ``scenario`` covariance is an ASSUMED
    sensitivity scenario (e.g. "every level uncertain by 0.001 cm^-1, independently"); it is not an experimentally
    established measurement covariance, and results propagated from it are labelled accordingly. Correlations are never
    inferred (not from a shared publication, not from a shared origin): they are either supplied or absent (diagonal).
    The arrays are read-only and cannot be re-enabled for writing; ``cov`` is the symmetrised, validated input."""
    names: tuple
    values: np.ndarray
    cov: np.ndarray
    units: tuple
    kind: str
    source: str

    def __post_init__(self):
        names = tuple(str(n) for n in self.names)
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("values must be a non-empty 1-D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("values contain non-finite entries")
        n = values.size
        if len(names) != n or len(set(names)) != n:
            raise ValueError("names must be unique and match the number of values")
        cov = np.asarray(self.cov, dtype=float)
        if cov.shape != (n, n):
            raise ValueError(f"covariance shape {cov.shape} does not match {n} values")
        validate_covariance(cov, name="measurement covariance")
        if self.kind not in PROVENANCE:
            raise ValueError(f"kind must be one of {PROVENANCE}, got {self.kind!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty provenance description")
        units = tuple(self.units) if not isinstance(self.units, str) else (self.units,) * n
        if len(units) != n:
            raise ValueError("units must be a string or one entry per value")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", _frozen(values))
        object.__setattr__(self, "cov", _frozen(0.5 * (cov + cov.T)))
        object.__setattr__(self, "units", tuple(str(u) for u in units))

    # --- constructors
    @classmethod
    def independent(cls, names, values, sigma, kind: str, source: str, units="") -> "MeasurementSet":
        """Independent standard uncertainties (diagonal covariance). ``sigma`` is a scalar or one value per entry, >= 0."""
        v = np.asarray(values, dtype=float)
        s = np.broadcast_to(np.asarray(sigma, dtype=float), v.shape).astype(float)
        if not np.all(np.isfinite(s)) or np.any(s < 0):
            raise ValueError("sigma must be finite and non-negative")
        return cls(tuple(names), v, np.diag(s ** 2), units, kind, source)

    @classmethod
    def from_correlation(cls, names, values, sigma, corr, kind: str, source: str, units="") -> "MeasurementSet":
        """Standard uncertainties plus an explicit correlation matrix R (symmetric, unit diagonal, entries in [-1, 1], PSD)."""
        v = np.asarray(values, dtype=float)
        s = np.broadcast_to(np.asarray(sigma, dtype=float), v.shape).astype(float)
        if not np.all(np.isfinite(s)) or np.any(s < 0):
            raise ValueError("sigma must be finite and non-negative")
        r = np.asarray(corr, dtype=float)
        if r.shape != (v.size, v.size):
            raise ValueError("correlation matrix shape does not match the values")
        validate_covariance(r, name="correlation matrix")
        if np.max(np.abs(np.diag(r) - 1.0)) > 1e-12 or np.max(np.abs(r)) > 1.0 + 1e-12:
            raise CovarianceError("correlation matrix must have unit diagonal and entries within [-1, 1]")
        return cls(tuple(names), v, r * np.outer(s, s), units, kind, source)

    @classmethod
    def from_covariance(cls, names, values, cov, kind: str, source: str, units="") -> "MeasurementSet":
        """A full covariance matrix (validated finite, symmetric, positive semidefinite)."""
        return cls(tuple(names), np.asarray(values, dtype=float), np.asarray(cov, dtype=float), units, kind, source)

    # --- views
    @property
    def n(self) -> int:
        return self.values.size

    @property
    def sigma(self) -> np.ndarray:
        return _frozen(np.sqrt(np.diag(self.cov)))

    @property
    def is_scenario(self) -> bool:
        return self.kind == "scenario"

    def correlation(self) -> np.ndarray:
        """Correlation matrix; entries involving a zero-variance input are NaN (undefined), the diagonal is 1 where defined."""
        s = np.sqrt(np.diag(self.cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.cov / np.outer(s, s)
        return _frozen(r)

    def check(self) -> CovarianceCheck:
        return validate_covariance(self.cov, name="measurement covariance")

    def describe(self) -> str:
        if self.is_scenario:
            return ("SCENARIO covariance (assumed; not an experimentally established measurement covariance): " + self.source)
        return f"{self.kind} covariance: " + self.source

    # --- positive-definite requirements (for a later weighted fit)
    def require_positive_definite(self) -> None:
        """Raise ``CovarianceError`` if the covariance is singular. A weighted fit needs C_y^-1 and a Cholesky whitening factor;
        this implementation neither regularises nor substitutes a pseudoinverse."""
        c = self.check()
        if not c.is_pd:
            raise CovarianceError(
                f"covariance is singular (rank {c.rank} of {c.n}); weighted fitting needs a positive-definite matrix. "
                "Singular covariances are valid for propagation (inverse_map_covariance) but are refused here; "
                "no regularisation or pseudoinverse is applied silently. Remove the redundant (perfectly correlated or "
                "zero-variance) combinations or reformulate the problem in independent variables.")

    def whitening_factor(self) -> np.ndarray:
        """Lower Cholesky factor L with C = L L^T (whitening is L^-1). Only for positive-definite C; raises otherwise."""
        self.require_positive_definite()
        return _frozen(cholesky(self.cov, lower=True))


# -------------------------------------------------------------------------------------------- numerical Jacobian
@dataclass(frozen=True)
class JacobianResult:
    jac: np.ndarray              # central difference at the base step
    steps: np.ndarray            # base step per input
    spread: float                # max relative difference between the Jacobians at steps h/2, h and 2h
    rtol: float
    stable: bool


def numerical_jacobian(func: Callable[[np.ndarray], np.ndarray], y, *, steps=None, rel_step: float = 1e-8,
                       step_rtol: float = 1e-5) -> JacobianResult:
    """Central-difference Jacobian A_ij = d func_i / d y_j with a step-stability check.

    The Jacobian is formed at steps h/2, h and 2h (h_j = ``steps[j]``, default ``rel_step * max(|y_j|, 1)``). The spread is
    the largest entry-wise difference between the three, relative to the largest magnitude in that column (so it is
    scale-free). ``stable`` is True when spread <= ``step_rtol``. A result is never accepted on the strength of one step.
    ``func`` must return a 1-D array; if it raises ``ValueError`` at a perturbed point a ``JacobianError`` is raised
    (the point is too close to the boundary of the map's domain for a local derivative)."""
    y = np.asarray(y, dtype=float)
    h = (np.asarray(steps, dtype=float) if steps is not None else rel_step * np.maximum(np.abs(y), 1.0))
    h = np.broadcast_to(h, y.shape).astype(float)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        raise JacobianError("finite-difference steps must be positive and finite")

    def eval_(x):
        try:
            out = np.atleast_1d(np.asarray(func(x), dtype=float))
        except ValueError as exc:
            raise JacobianError(f"the map is undefined at a perturbed point ({exc}); the point is too close to the boundary "
                                "of its domain for a local derivative") from exc
        if out.ndim != 1 or not np.all(np.isfinite(out)):
            raise JacobianError("the map returned non-finite values or a non-1-D result")
        return out

    f0 = eval_(y)
    jacs = []
    for mult in (0.5, 1.0, 2.0):
        a = np.empty((f0.size, y.size))
        for j in range(y.size):
            e = np.zeros_like(y)
            e[j] = mult * h[j]
            a[:, j] = (eval_(y + e) - eval_(y - e)) / (2.0 * mult * h[j])
        jacs.append(a)
    stack = np.stack(jacs)                                    # (3, m, n)
    col_scale = np.max(np.abs(stack), axis=(0, 1))            # per input column
    diff = np.max(stack, axis=0) - np.min(stack, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(col_scale > 0, np.max(diff, axis=0) / col_scale, 0.0)
    spread = float(np.max(rel)) if rel.size else 0.0
    return JacobianResult(jac=_frozen(jacs[1]), steps=_frozen(h), spread=spread, rtol=step_rtol, stable=bool(spread <= step_rtol))


# ------------------------------------------------------------------------------------------------ identifiability
@dataclass(frozen=True)
class WeakDirection:
    singular_value: float
    ratio_to_largest: float
    components: dict             # parameter name -> component of the (unit) weak parameter combination, largest first


@dataclass(frozen=True)
class Identifiability:
    """SVD diagnostics of a whitened, parameter-scaled Jacobian. See ``identifiability`` for the exact definition.

    ``condition_number`` is s_max / s_min of the matrix defined by ``scaling``; it is meaningful only together with that
    scaling convention and says nothing by itself about physical non-identifiability."""
    orientation: str
    scaling: str
    singular_values: np.ndarray
    rank: int
    rank_tol: float
    n_params: int
    condition_number: float
    weak_directions: tuple
    max_abs_correlation: float | None
    flag_cond: float
    refuse_cond: float
    status: str                  # 'ok' | 'flagged' | 'refused'
    messages: tuple

    def summary(self) -> str:
        return (f"{self.status.upper()}: rank {self.rank}/{self.n_params}, cond = {self.condition_number:.3g} "
                f"[{self.scaling}]" + ("; " + "; ".join(self.messages) if self.messages else ""))


def identifiability(jac, *, cov_y=None, scales=None, orientation: str = "forward", param_names: Sequence[str] | None = None,
                    expect_independent_params: bool = True, rank_rtol: float | None = None, flag_cond: float = 1e6,
                    refuse_cond: float = 1e8, correlation=None, corr_flag: float = 0.99, n_weak: int = 3,
                    scales_note: str | None = None) -> Identifiability:
    r"""SVD diagnostics of a scaled Jacobian. The thresholds are configurable diagnostic conventions, not physical limits.

    ``orientation="forward"``: ``jac`` = df/dp (n_obs x m). With ``cov_y = L L^T`` (Cholesky; must be positive definite) the
        matrix analysed is  J_scaled = L^-1 J S_p,  S_p = diag(``scales``) the characteristic parameter scales.
        Without ``cov_y`` the observations are not whitened (J S_p), and the scaling string says so.
        Weak combinations are right singular vectors (directions in scaled parameter space).
    ``orientation="inverse"``: ``jac`` = A = dp/dy (m x n_in), the Jacobian of an explicit map p = p(y).
        The matrix analysed is  A_scaled = S_p^-1 A D_y,  D_y = diag(sqrt(diag(cov_y))) (input standard deviations; no
        full whitening is needed, and singular covariances are allowed). Weak combinations are left singular vectors.

    ``scales`` (length m) are the characteristic parameter scales (default 1, stated in the scaling string).
    Numerical rank: singular values above ``rank_rtol * s_max`` (default max(shape) * machine epsilon).
    Status: ``refused`` if the matrix is not finite, the rank is below the number of parameters while
    ``expect_independent_params`` is True, or cond >= ``refuse_cond``; ``flagged`` if cond >= ``flag_cond``, the rank is below
    the number of parameters for outputs that are allowed to be dependent, or the largest |correlation| (if ``correlation`` is
    given) exceeds ``corr_flag``; otherwise ``ok``. No regularisation is applied."""
    a = np.asarray(jac, dtype=float)
    if a.ndim != 2:
        raise ValueError("jac must be a 2-D array")
    if orientation not in ("forward", "inverse"):
        raise ValueError("orientation must be 'forward' or 'inverse'")
    m = a.shape[1] if orientation == "forward" else a.shape[0]
    names = tuple(param_names) if param_names is not None else tuple(f"p{i}" for i in range(m))
    if len(names) != m:
        raise ValueError("param_names length does not match the number of parameters")
    if scales is None:
        s_p, sc_txt = np.ones(m), "parameter scales S_p = 1 (unscaled)"
    else:
        s_p = np.asarray(scales, dtype=float)
        if s_p.shape != (m,) or np.any(s_p <= 0) or not np.all(np.isfinite(s_p)):
            raise ValueError("scales must be positive, finite, one per parameter")
        sc_txt = scales_note or "parameter scales S_p given explicitly"
    if not np.all(np.isfinite(a)):
        return Identifiability(orientation, sc_txt, np.array([]), 0, float("nan"), m, float("inf"), (), None, flag_cond, refuse_cond,
                               "refused", ("the Jacobian contains non-finite values",))
    if orientation == "forward":
        if cov_y is not None:
            c = np.asarray(cov_y, dtype=float)
            ck = validate_covariance(c, name="cov_y")
            if not ck.is_pd:
                raise CovarianceError("forward identifiability with whitening needs a positive-definite cov_y "
                                      f"(rank {ck.rank} of {ck.n}); singular matrices are not regularised")
            low = cholesky(c, lower=True)
            mat = solve_triangular(low, a, lower=True) * s_p[None, :]
            sc_txt += "; observations whitened, J_scaled = L^-1 J S_p with C_y = L L^T"
        else:
            mat = a * s_p[None, :]
            sc_txt += "; observations NOT whitened (J S_p)"
    else:
        if cov_y is not None:
            c = np.asarray(cov_y, dtype=float)
            validate_covariance(c, name="cov_y")
            d_y = np.sqrt(np.diag(c))
            sc_txt += "; inputs scaled by their standard deviations, A_scaled = S_p^-1 A diag(sigma_y)"
        else:
            d_y = np.ones(a.shape[1])
            sc_txt += "; inputs NOT scaled (S_p^-1 A)"
        mat = a / s_p[:, None] * d_y[None, :]
    u, sv, vt = np.linalg.svd(mat, full_matrices=True)
    s_max = float(sv[0]) if sv.size else 0.0
    tol = (rank_rtol if rank_rtol is not None else max(mat.shape) * _EPS) * s_max
    rank = int(np.sum(sv > tol))
    cond = float(s_max / sv[m - 1]) if (rank == m and m > 0) else float("inf")   # infinite when the matrix is rank-deficient
    msgs = []
    status = "ok"
    if s_max == 0.0:
        status, msgs = "refused", ["the scaled Jacobian is identically zero"]
    else:
        if rank < m:
            if expect_independent_params:
                status = "refused"
                msgs.append(f"rank {rank} < {m} parameters: some parameter combination has no sensitivity to the data (not identifiable)")
            else:
                status = "flagged"
                msgs.append(f"rank {rank} < {m} outputs: the outputs are functionally dependent (singular output covariance)")
        elif cond >= refuse_cond:
            status = "refused"
            msgs.append(f"condition number {cond:.3g} >= refuse threshold {refuse_cond:.3g} for this scaling")
        elif cond >= flag_cond:
            status = "flagged"
            msgs.append(f"condition number {cond:.3g} >= flag threshold {flag_cond:.3g} for this scaling")
    # weak combinations: smallest singular values among the m directions
    weak = []
    if s_max > 0:
        vecs = vt.T if orientation == "forward" else u                    # columns live in scaled parameter space (length m)
        order = list(range(min(m, sv.size)))[::-1]                          # smallest first
        for idx in order[:n_weak]:
            ratio = float(sv[idx] / s_max)
            if ratio * flag_cond < 1.0 or idx >= rank:
                comp = vecs[:, idx]
                top = np.argsort(-np.abs(comp))[: min(3, m)]
                weak.append(WeakDirection(float(sv[idx]), ratio, {names[k]: float(comp[k]) for k in top}))
        for k in range(sv.size, m):                                         # null directions when the matrix is short
            comp = vecs[:, k]
            top = np.argsort(-np.abs(comp))[: min(3, m)]
            weak.append(WeakDirection(0.0, 0.0, {names[j]: float(comp[j]) for j in top}))
    max_corr = None
    if correlation is not None:
        r = np.asarray(correlation, dtype=float)
        off = np.abs(r - np.diag(np.diag(r)))
        off = off[np.isfinite(off)]
        max_corr = float(off.max()) if off.size else 0.0
        if max_corr > corr_flag and expect_independent_params and status == "ok":
            status = "flagged"
            msgs.append(f"largest |correlation| {max_corr:.4g} exceeds {corr_flag}")
    return Identifiability(orientation, sc_txt, _frozen(sv), rank, float(tol), m, cond, tuple(weak), max_corr, flag_cond, refuse_cond,
                           status, tuple(msgs))


# ---------------------------------------------------------------------------------------------- inverse-map covariance
@dataclass(frozen=True)
class InverseMapResult:
    """Output covariance C_p = A C_y A^T of an explicit map p = p(y), to first order.

    ``covariance``/``correlation`` are None when ``status == 'refused'`` (the reasons are in ``messages``).
    ``uncertainty_kind`` is the provenance of the INPUT covariance; ``is_scenario`` is True when it was an assumed scenario,
    in which case ``label`` says the outputs are a sensitivity scenario and not an experimental uncertainty."""
    names: tuple
    units: tuple
    values: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray | None
    correlation: np.ndarray | None
    std: np.ndarray | None
    uncertainty_kind: str
    is_scenario: bool
    label: str
    status: str
    messages: tuple
    identifiability: Identifiability
    jacobian_step_spread: float | None
    jacobian_source: str
    input_names: tuple = field(default_factory=tuple)


def inverse_map_covariance(param_map: Callable[[np.ndarray], np.ndarray], inputs: MeasurementSet, *, output_names=None,
                           output_units=None, jacobian=None, param_scales=None, outputs_are_parameters: bool = True,
                           steps=None, rel_step: float = 1e-8, step_rtol: float = 1e-5, flag_cond: float = 1e6,
                           refuse_cond: float = 1e8, corr_flag: float = 0.99) -> InverseMapResult:
    r"""First-order covariance of p = param_map(y) for inputs y with covariance C_y:  C_p = A C_y A^T,  A = dp/dy.

    ``param_map(y)`` takes the input vector (ordered as ``inputs.names``) and returns a 1-D array. ``jacobian`` may be an
    (m x n) array or a callable y -> array (analytical derivatives, used as given); otherwise A is computed by
    ``numerical_jacobian``, which must pass its step-stability check or the result is ``refused`` (no covariance returned).
    ``C_y`` may be singular (propagation is valid for any positive-semidefinite C_y).

    ``outputs_are_parameters=True`` means the m outputs are meant to be independent parameters of a calibration: a Jacobian of rank
    below m (or an extreme condition number) then REFUSES the result instead of returning a misleading covariance. Use ``False``
    when some outputs are derived from the others (e.g. calcium eps^2 next to its four parameters): rank deficiency is then
    only flagged, and the (singular) output covariance is returned.

    ``param_scales`` are the characteristic scales S_p used by the identifiability diagnostic (default |p_hat|, or 1 where p_hat = 0).
    The diagnostic matrix is S_p^-1 A diag(sigma_y); the scaling is recorded in ``identifiability.scaling``.
    The result is a conditional, first-order statement; it contains no model error and is not a goodness-of-fit."""
    y0 = np.asarray(inputs.values, dtype=float)
    p0 = np.atleast_1d(np.asarray(param_map(y0), dtype=float))
    if p0.ndim != 1 or not np.all(np.isfinite(p0)):
        raise ValueError("param_map must return a finite 1-D array")
    m = p0.size
    names = tuple(output_names) if output_names is not None else tuple(f"p{i}" for i in range(m))
    units = tuple(output_units) if output_units is not None else ("",) * m
    if len(names) != m or len(units) != m:
        raise ValueError("output_names/output_units must have one entry per output")
    msgs: list[str] = []
    spread = None
    if jacobian is not None:
        a = np.asarray(jacobian(y0) if callable(jacobian) else jacobian, dtype=float)
        src = "analytical (supplied)"
        if a.shape != (m, inputs.n):
            raise JacobianError(f"supplied Jacobian has shape {a.shape}, expected {(m, inputs.n)}")
        if not np.all(np.isfinite(a)):
            raise JacobianError("supplied Jacobian contains non-finite values")
        stable = True
    else:
        jr = numerical_jacobian(lambda x: param_map(x), y0, steps=steps, rel_step=rel_step, step_rtol=step_rtol)
        a, spread, stable = np.array(jr.jac), jr.spread, jr.stable
        src = f"numerical central difference (h = {rel_step:g}*max(|y|,1), checked at h/2, h, 2h)"
        if not stable:
            msgs.append(f"finite-difference Jacobian is not step-stable (spread {spread:.3g} > {step_rtol:g}); no covariance returned")
    scales = (np.where(np.abs(p0) > 0, np.abs(p0), 1.0) if param_scales is None else np.asarray(param_scales, dtype=float))
    scales_note = ("parameter scales S_p = |p_hat| (default; 1 where p_hat = 0)" if param_scales is None
                   else "parameter scales S_p supplied by the caller")
    idn = identifiability(a, cov_y=inputs.cov, scales=scales, orientation="inverse", param_names=names,
                          expect_independent_params=outputs_are_parameters, flag_cond=flag_cond, refuse_cond=refuse_cond,
                          corr_flag=corr_flag, scales_note=scales_note)
    cov = corr = std = None
    status = idn.status
    msgs.extend(idn.messages)
    if not stable:
        status = "refused"
    if status != "refused":
        cov_p = a @ inputs.cov @ a.T
        cov_p = 0.5 * (cov_p + cov_p.T)
        std = np.sqrt(np.maximum(np.diag(cov_p), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov_p / np.outer(std, std)
        cov = cov_p
        if idn.max_abs_correlation is None:
            off = np.abs(corr - np.diag(np.diag(corr)))
            off = off[np.isfinite(off)]
            mc = float(off.max()) if off.size else 0.0
            idn = Identifiability(**{**idn.__dict__, "max_abs_correlation": mc})
            if mc > corr_flag and outputs_are_parameters and status == "ok":
                status = "flagged"
                msgs.append(f"largest |correlation| {mc:.4g} exceeds {corr_flag}")
    scen = inputs.is_scenario
    label = ("SENSITIVITY SCENARIO propagated from an assumed input covariance (not an experimental uncertainty); first-order, "
             "conditional on the map; excludes model error" if scen else
             f"first-order propagation of {inputs.kind} input covariance; conditional on the map; excludes model error and "
             "is not a goodness-of-fit")
    return InverseMapResult(names=names, units=units, values=_frozen(p0), jacobian=_frozen(a),
                            covariance=None if cov is None else _frozen(cov), correlation=None if corr is None else _frozen(corr),
                            std=None if std is None else _frozen(std), uncertainty_kind=inputs.kind, is_scenario=scen, label=label,
                            status=status, messages=tuple(msgs), identifiability=idn, jacobian_step_spread=spread,
                            jacobian_source=src, input_names=inputs.names)
