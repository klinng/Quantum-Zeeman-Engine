r"""Core covariance infrastructure (Stage 8B.1): measurement covariance, inverse-map propagation, identifiability.

WHAT IS IMPLEMENTED
-------------------
* ``MeasurementSet``: named values with units, a validated covariance matrix and an explicit provenance
  (``measured``, ``theory``, ``recommended`` or ``scenario``). The arrays are read-only.
* ``validate_covariance``: finite values, symmetry and positive-semidefiniteness with scale-aware tolerances.
* ``numerical_jacobian``: central differences at three step sizes with a step-stability check.
* ``inverse_map_covariance``: C_p = A C_y A^T for an explicit parameter map p = p(y), A = dp/dy.
* ``identifiability``: STRUCTURAL identifiability, SVD of a Jacobian scaled by explicit characteristic scales (rank,
  condition number, weak combinations, status). For an inverse map it is A_struct = S_p^-1 A S_y and does not depend on the
  input covariance, so zero input uncertainty is not a structural failure.
* ``uncertainty_sensitivity``: separate, informational diagnostic of which parameter combinations respond to the supplied
  uncertainty: marginal (S_p^-1 A diag(sigma_y), ignores correlations) and covariance-aware (S_p^-1 A R, C_y = R R^T with a
  PSD square root R). It never refuses. Squared covariance-aware singular values are the eigenvalues of S_p^-1 C_p S_p^-1.

* (Stage 8B.2a) ``fit_parameter_covariance`` / ``FitCovariance``: a PURE function that turns a model Jacobian J = d(model)/dp
  at a solution, plus an explicitly supplied observation covariance or the residuals, into a parameter covariance, in one of
  four explicitly selected modes: ``sandwich`` (unweighted estimator, known C_y), ``weighted`` (estimator that minimised
  chi^2 with the same C_y), ``residual_variance`` (s^2 (J^T J)^-1, opt-in) and ``exact`` (square J, dof = 0). It never
  touches an estimator; no fitting routine calls it yet.
* (Stage 8B.2a) per-element provenance in ``MeasurementSet`` (``kind`` may be one string or one string per element).

WHAT IS NOT IMPLEMENTED (deliberately)
--------------------------------------
A general Monte Carlo engine, helium numerical covariance, calcium wrappers, GLS or residual-variance INTEGRATION into
``fit_parameters`` / ``fit_to_levels``, and any PyTorch route. No pseudoinverse, regularisation, default uncertainty or
automatic noise estimate is used anywhere.

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


def _check_permutation(order, n: int) -> np.ndarray:
    o = np.asarray(order)
    if o.ndim != 1 or o.size != n or not np.issubdtype(o.dtype, np.integer) or sorted(o.tolist()) != list(range(n)):
        raise ValueError(f"order must be a permutation of range({n})")
    return o.astype(int)


# ------------------------------------------------------------------------------------------------ MeasurementSet
@dataclass(frozen=True)
class MeasurementSet:
    """Named measured (or assumed) values with a covariance matrix and an explicit provenance.

    ``kind`` is one of ``measured``, ``theory``, ``recommended``, ``scenario``, or a sequence of one such string per element
    (per-element provenance, e.g. one measured interval among assumed levels). ``element_kinds`` always holds the per-element
    tuple; ``kind`` is the common value when all elements agree and ``"mixed"`` otherwise. ``is_scenario`` is True when ANY
    element is a scenario, so every result propagated from a set containing an assumed element keeps its scenario label.
    A ``scenario`` covariance is an ASSUMED
    sensitivity scenario (e.g. "every level uncertain by 0.001 cm^-1, independently"); it is not an experimentally
    established measurement covariance, and results propagated from it are labelled accordingly. Correlations are never
    inferred (not from a shared publication, not from a shared origin): they are either supplied or absent (diagonal).
    The arrays are read-only and cannot be re-enabled for writing; ``cov`` is the symmetrised, validated input."""
    names: tuple
    values: np.ndarray
    cov: np.ndarray
    units: tuple
    kind: "str | Sequence[str]"
    source: str
    element_kinds: tuple = field(default=(), init=False)

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
        if isinstance(self.kind, str):
            if self.kind not in PROVENANCE:
                raise ValueError(f"kind must be one of {PROVENANCE}, got {self.kind!r}")
            elem = (self.kind,) * n
        else:
            elem = tuple(self.kind)
            if len(elem) != n:
                raise ValueError(f"per-element kind needs one entry per value ({n}), got {len(elem)}")
            bad = [k for k in elem if k not in PROVENANCE]
            if bad:
                raise ValueError(f"every element kind must be one of {PROVENANCE}, got {bad!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty provenance description")
        units = tuple(self.units) if not isinstance(self.units, str) else (self.units,) * n
        if len(units) != n:
            raise ValueError("units must be a string or one entry per value")
        object.__setattr__(self, "element_kinds", elem)
        object.__setattr__(self, "kind", elem[0] if len(set(elem)) == 1 else "mixed")
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
        """True if ANY element is an assumed scenario (the conservative label for anything propagated from this set)."""
        return "scenario" in self.element_kinds

    @property
    def is_mixed(self) -> bool:
        return self.kind == "mixed"

    @property
    def scenario_elements(self) -> tuple:
        return tuple(nm for nm, k in zip(self.names, self.element_kinds) if k == "scenario")

    def kinds_by_name(self) -> dict:
        return dict(zip(self.names, self.element_kinds))

    def reordered(self, order) -> "MeasurementSet":
        """A new set whose entry i is entry ``order[i]`` of this one (values, names, units, per-element kinds and BOTH the rows and
        columns of the covariance are permuted together). ``order`` must be a permutation of range(n). Use it to align a covariance
        with the observation ordering of a model, e.g. ``order = np.argsort(values)`` for a model that returns sorted levels."""
        o = _check_permutation(order, self.n)
        return MeasurementSet(tuple(self.names[i] for i in o), self.values[o], self.cov[np.ix_(o, o)],
                              tuple(self.units[i] for i in o), tuple(self.element_kinds[i] for i in o), self.source)

    def correlation(self) -> np.ndarray:
        """Correlation matrix; entries involving a zero-variance input are NaN (undefined), the diagonal is 1 where defined."""
        s = np.sqrt(np.diag(self.cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.cov / np.outer(s, s)
        return _frozen(r)

    def check(self) -> CovarianceCheck:
        return validate_covariance(self.cov, name="measurement covariance")

    def describe(self) -> str:
        if self.is_mixed:
            groups = {}
            for nm, k in zip(self.names, self.element_kinds):
                groups.setdefault(k, []).append(nm)
            txt = "; ".join(f"{k}: {', '.join(v)}" for k, v in groups.items())
            head = ("MIXED provenance, SCENARIO-dependent (the assumed elements are not experimentally established)"
                    if self.is_scenario else "MIXED provenance")
            return f"{head} [{txt}]: " + self.source
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
                    scales_note: str | None = None, input_scales=None, input_scales_note: str | None = None) -> Identifiability:
    r"""STRUCTURAL identifiability: SVD diagnostics of a Jacobian scaled by explicit characteristic scales. The thresholds are
    configurable diagnostic conventions, not physical limits. Nothing here depends on the supplied input covariance of an
    inverse map: a zero or tiny uncertainty never makes a well-defined map unidentifiable (see ``uncertainty_sensitivity`` for
    the separate, covariance-dependent diagnostic).

    ``orientation="forward"``: ``jac`` = df/dp (n_obs x m). With ``cov_y = L L^T`` (Cholesky; must be positive definite) the
        matrix analysed is  J_scaled = L^-1 J S_p,  S_p = diag(``scales``) the characteristic parameter scales.
        Without ``cov_y`` the observations are not whitened (J S_p), and the scaling string says so.
        Weak combinations are right singular vectors (directions in scaled parameter space).
    ``orientation="inverse"``: ``jac`` = A = dp/dy (m x n_in), the Jacobian of an explicit map p = p(y).
        The matrix analysed is  A_struct = S_p^-1 A S_y  with S_y = diag(``input_scales``) the characteristic INPUT scales
        (default 1, stated in the scaling string). ``cov_y`` is not accepted: structural identifiability does not depend on it.
        Meaning of the SVD A_struct = U diag(s) V^T: s_i are the gains from relative input changes (V columns, combinations of
        the inputs in units of S_y) to relative parameter changes (U columns, combinations of the parameters in units of
        S_p); rank < m means some parameter combination does not respond to any input (the parameters are functionally
        dependent); s_max/s_min (``condition_number``) measures how unevenly the parameter combinations respond, in these
        characteristic units. Weak combinations are left singular vectors (U columns).

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
            raise ValueError("structural identifiability of an inverse map does not use cov_y; pass input_scales, and use "
                             "uncertainty_sensitivity for the covariance-dependent diagnostic")
        if input_scales is None:
            s_y, sy_txt = np.ones(a.shape[1]), "input scales S_y = 1 (unscaled)"
        else:
            s_y = np.asarray(input_scales, dtype=float)
            if s_y.shape != (a.shape[1],) or np.any(s_y <= 0) or not np.all(np.isfinite(s_y)):
                raise ValueError("input_scales must be positive, finite, one per input")
            sy_txt = input_scales_note or "input scales S_y given explicitly"
        sc_txt += f"; {sy_txt}; A_struct = S_p^-1 A S_y (independent of the input covariance)"
        mat = a / s_p[:, None] * s_y[None, :]
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


# ---------------------------------------------------------------------------- uncertainty-scaled sensitivity
@dataclass(frozen=True)
class UncertaintySensitivity:
    """Which parameter combinations respond to the SUPPLIED input uncertainty. This is not a structural diagnostic: a zero or
    small value here means "no/little propagated uncertainty", never "unidentifiable".

    ``marginal_*``: SVD of S_p^-1 A diag(sigma_y); uses only the marginal standard deviations and IGNORES input correlations.
    ``covariance_*``: SVD of S_p^-1 A R with C_y = R R^T, R = V sqrt(max(lambda, 0)) from the symmetric eigendecomposition
    C_y = V diag(lambda) V^T (a positive-semidefinite square root; singular C_y is allowed). It uses the full covariance; its
    squared singular values are the eigenvalues of the scaled output covariance S_p^-1 C_p S_p^-1, and its left singular vectors
    are the principal axes of the propagated uncertainty in units of S_p. ``covariance_cond`` is s_max/s_min of that matrix
    (inf when the propagated covariance is singular, nan when it is zero)."""
    marginal_description: str
    covariance_description: str
    marginal_singular_values: np.ndarray
    covariance_singular_values: np.ndarray
    covariance_rank: int
    covariance_cond: float
    is_zero: bool
    weak_directions: tuple


def uncertainty_sensitivity(jac, cov_y, *, scales=None, param_names: Sequence[str] | None = None, rank_rtol: float | None = None,
                            n_weak: int = 3) -> UncertaintySensitivity:
    """Uncertainty-scaled sensitivity of an explicit map p = p(y) with Jacobian ``jac`` = A (m x n) and input covariance
    ``cov_y`` (any positive-semidefinite matrix, including zero). See ``UncertaintySensitivity`` for the matrices analysed.
    ``weak_directions`` lists up to ``n_weak`` left singular vectors (parameter-space directions, smallest singular value first)
    with a rank-deficient or negligible (< 1e-6 relative) response; when there are more outputs than inputs (m > n) the exact
    null-space directions beyond min(m, n) are included with singular value 0. Raises ValueError for a non-2-D or non-finite
    Jacobian, a ``param_names`` length different from m, or invalid ``scales``; CovarianceError for an invalid ``cov_y``."""
    a = np.asarray(jac, dtype=float)
    if a.ndim != 2:
        raise ValueError("jac must be a 2-D array")
    if not np.all(np.isfinite(a)):
        raise ValueError("jac contains non-finite values")
    m, n = a.shape
    c = np.asarray(cov_y, dtype=float)
    if c.shape != (n, n):
        raise ValueError("cov_y must be n x n for an m x n Jacobian")
    validate_covariance(c, name="cov_y")
    s_p = np.ones(m) if scales is None else np.asarray(scales, dtype=float)
    if s_p.shape != (m,) or np.any(s_p <= 0) or not np.all(np.isfinite(s_p)):
        raise ValueError("scales must be positive, finite, one per parameter")
    names = tuple(param_names) if param_names is not None else tuple(f"p{i}" for i in range(m))
    if len(names) != m:
        raise ValueError(f"param_names has {len(names)} entries but the Jacobian has {m} parameters (rows)")
    lam, vec = np.linalg.eigh(0.5 * (c + c.T))
    root = vec * np.sqrt(np.maximum(lam, 0.0))[None, :]                 # C_y = root root^T
    mat_marg = a / s_p[:, None] * np.sqrt(np.diag(c))[None, :]
    mat_cov = a / s_p[:, None] @ root
    sv_m = np.linalg.svd(mat_marg, compute_uv=False)
    u, sv, _ = np.linalg.svd(mat_cov, full_matrices=True)
    s_max = float(sv[0]) if sv.size else 0.0
    tol = (rank_rtol if rank_rtol is not None else max(mat_cov.shape) * _EPS) * s_max
    rank = int(np.sum(sv > tol)) if s_max > 0 else 0
    if s_max == 0.0:
        cond = float("nan")
    else:
        cond = float(s_max / sv[m - 1]) if rank == m else float("inf")
    weak = []
    if s_max > 0:
        for idx in range(m - 1, -1, -1):                                  # smallest first; u is m x m, so when m > n the
            sval = float(sv[idx]) if idx < sv.size else 0.0              # directions beyond min(m, n) are exact null directions
            if idx >= rank or sval / s_max < 1e-6:
                comp = u[:, idx]
                top = np.argsort(-np.abs(comp))[: min(3, m)]
                weak.append(WeakDirection(sval, sval / s_max, {names[k]: float(comp[k]) for k in top}))
                if len(weak) >= n_weak:
                    break
    return UncertaintySensitivity(
        "S_p^-1 A diag(sigma_y): marginal standard deviations only; input correlations are NOT included",
        "S_p^-1 A R with C_y = R R^T (PSD square root): full input covariance including correlations",
        _frozen(sv_m), _frozen(sv), rank, cond, bool(s_max == 0.0), tuple(weak))


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
    sensitivity: UncertaintySensitivity | None = None


def inverse_map_covariance(param_map: Callable[[np.ndarray], np.ndarray], inputs: MeasurementSet, *, output_names=None,
                           output_units=None, jacobian=None, param_scales=None, outputs_are_parameters: bool = True,
                           steps=None, rel_step: float = 1e-8, step_rtol: float = 1e-5, flag_cond: float = 1e6,
                           refuse_cond: float = 1e8, corr_flag: float = 0.99,
                           input_scales=None) -> InverseMapResult:
    r"""First-order covariance of p = param_map(y) for inputs y with covariance C_y:  C_p = A C_y A^T,  A = dp/dy.

    ``param_map(y)`` takes the input vector (ordered as ``inputs.names``) and returns a 1-D array. ``jacobian`` may be an
    (m x n) array or a callable y -> array (analytical derivatives, used as given); otherwise A is computed by
    ``numerical_jacobian``, which must pass its step-stability check or the result is ``refused`` (no covariance returned).
    ``C_y`` may be singular (propagation is valid for any positive-semidefinite C_y).

    ``outputs_are_parameters=True`` means the m outputs are meant to be independent parameters of a calibration: a Jacobian of rank
    below m (or an extreme condition number) then REFUSES the result instead of returning a misleading covariance. Use ``False``
    when some outputs are derived from the others (e.g. calcium eps^2 next to its four parameters): rank deficiency is then
    only flagged, and the (singular) output covariance is returned.

    Two separate diagnostics are returned. (1) ``identifiability`` is STRUCTURAL: A_struct = S_p^-1 A S_y with explicit
    characteristic parameter scales S_p (``param_scales``, default |p_hat|, 1 where p_hat = 0) and input scales S_y
    (``input_scales``, default |y_hat|, 1 where y_hat = 0); it does not depend on C_y, so a zero input covariance gives the
    zero output covariance rather than a refusal. It alone decides refusal. (2) ``sensitivity`` is the uncertainty-scaled,
    informational diagnostic (marginal and covariance-aware); it never refuses.
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
    in_scales = (np.where(np.abs(y0) > 0, np.abs(y0), 1.0) if input_scales is None else np.asarray(input_scales, dtype=float))
    in_note = ("input scales S_y = |y_hat| (default; 1 where y_hat = 0)" if input_scales is None
               else "input scales S_y supplied by the caller")
    scales_note = ("parameter scales S_p = |p_hat| (default; 1 where p_hat = 0)" if param_scales is None
                   else "parameter scales S_p supplied by the caller")
    idn = identifiability(a, scales=scales, orientation="inverse", param_names=names, input_scales=in_scales,
                          input_scales_note=in_note, expect_independent_params=outputs_are_parameters, flag_cond=flag_cond,
                          refuse_cond=refuse_cond, corr_flag=corr_flag, scales_note=scales_note)
    sens = uncertainty_sensitivity(a, inputs.cov, scales=scales, param_names=names)
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
    label = (("SENSITIVITY SCENARIO propagated from an assumed input covariance (not an experimental uncertainty); first-order, "
              "conditional on the map; excludes model error"
              + ("; the inputs have MIXED provenance, scenario elements: " + ", ".join(inputs.scenario_elements)
                 if inputs.is_mixed else "")) if scen else
             f"first-order propagation of {inputs.kind} input covariance; conditional on the map; excludes model error and "
             "is not a goodness-of-fit")
    return InverseMapResult(names=names, units=units, values=_frozen(p0), jacobian=_frozen(a),
                            covariance=None if cov is None else _frozen(cov), correlation=None if corr is None else _frozen(corr),
                            std=None if std is None else _frozen(std), uncertainty_kind=inputs.kind, is_scenario=scen, label=label,
                            status=status, messages=tuple(msgs), identifiability=idn, jacobian_step_spread=spread,
                            jacobian_source=src, input_names=inputs.names, sensitivity=sens)


# =============================================================================== parameter covariance of a fit (Stage 8B.2a)
FIT_COVARIANCE_MODES = ("sandwich", "weighted", "residual_variance", "exact")
UNAVAILABLE_KINDS = ("not_computed", "not_applicable", "not_available", "refused")


@dataclass(frozen=True)
class FitCovariance:
    """Parameter covariance of a fit, or an honest statement that it is unavailable.

    ``covariance is None`` means UNAVAILABLE (``kind`` in ``UNAVAILABLE_KINDS`` and ``reason`` says why); it is never replaced by
    a zero matrix. A mathematically valid zero covariance (e.g. exactly known inputs) is an all-zero array with
    ``status == "ok"``. ``kind`` of an available result is the mode that produced it (``sandwich``, ``weighted``,
    ``residual_variance``, ``exact``). ``provenance`` is the input-covariance provenance (``scenario`` if any element is an
    assumed scenario, ``mixed`` for mixed sets, ``residual_scaled`` for the residual-variance mode, ``unspecified`` for a bare
    array); ``is_scenario`` keeps the scenario qualification. ``label`` states what the number is and is not. The covariance is
    conditional on the model and first order; it contains no model discrepancy and is not a goodness-of-fit.
    Ordering contract: parameter order = columns of the Jacobian (``param_names``); observation order = rows of the Jacobian
    (``observation_names`` when supplied)."""
    covariance: np.ndarray | None
    std: np.ndarray | None
    correlation: np.ndarray | None
    kind: str
    status: str                       # 'ok' | 'flagged' | 'refused' | 'unavailable'
    reason: str
    provenance: str
    is_scenario: bool
    label: str
    param_names: tuple
    observation_names: tuple | None
    n_obs: int | None
    n_params: int | None
    dof: int | None
    rss: float | None = None          # residual sum of squares in the units of the residuals (reported, not used unless mode says so)
    s2: float | None = None           # residual variance RSS/dof (residual_variance mode only)
    s2_relative_uncertainty: float | None = None   # about 1/sqrt(2 dof): the scatter of s^2 itself
    chi2: float | None = None         # weighted mode with residuals only; reported, never used to rescale the covariance
    chi2_per_dof: float | None = None
    identifiability: "Identifiability | None" = None
    messages: tuple = ()

    @property
    def available(self) -> bool:
        return self.covariance is not None

    @classmethod
    def unavailable(cls, kind: str, reason: str, *, status: str | None = None, param_names=(), provenance: str = "none",
                    is_scenario: bool = False, n_obs=None, n_params=None, dof=None, identifiability=None, messages=()):
        """An explicit 'no covariance' result. ``kind``: not_computed (nobody asked), not_applicable (e.g. the residual is model
        discrepancy, not noise), not_available (no defensible uncertainties exist), refused (asked for, but not identifiable or
        impossible)."""
        if kind not in UNAVAILABLE_KINDS:
            raise ValueError(f"kind must be one of {UNAVAILABLE_KINDS}")
        if not reason or not str(reason).strip():
            raise ValueError("an unavailable covariance needs a reason")
        return cls(None, None, None, kind, status or ("refused" if kind == "refused" else "unavailable"), str(reason), provenance,
                   is_scenario, f"NO COVARIANCE ({kind}): {reason}", tuple(param_names), None, n_obs, n_params, dof,
                   identifiability=identifiability, messages=tuple(messages))


def fit_parameter_covariance(jac, *, mode: str, cov_y=None, residuals=None, param_names: Sequence[str] | None = None,
                             param_values=None, scales=None, observation_names: Sequence[str] | None = None,
                             measurement_order=None, objective_is_weighted: bool = False, flag_cond: float = 1e6,
                             refuse_cond: float = 1e8, rank_rtol: float | None = None) -> FitCovariance:
    r"""Parameter covariance of a fit from the model Jacobian at the solution. Pure: no estimator is run or changed.

    ``jac`` = J = d(model prediction_i)/d p_j, shape (N observations x p parameters), in the units of the observations per unit of
    the parameters, evaluated at the fitted parameters (central differences with ``numerical_jacobian`` are advised). The
    covariance is first order (Gauss-Newton) and conditional on the model being right. NO pseudoinverse and NO regularisation is
    used: a rank-deficient or ill-conditioned J (cond >= ``refuse_cond``) is REPORTED and refused (covariance None).

    Modes (the caller selects one; nothing is inferred):

    ``"sandwich"`` (needs ``cov_y``, any positive-semidefinite matrix including zero): for the UNWEIGHTED estimator
        p_hat = argmin |r|^2 the linearised estimator is dp = A dy with A = (J^T J)^-1 J^T, so
        C_p = A C_y A^T = (J^T J)^-1 J^T C_y J (J^T J)^-1.  A is the exact left inverse of a full-column-rank J (QR), not a
        pseudoinverse of a deficient matrix. Equals (J^T C_y^-1 J)^-1 only when C_y is proportional to the identity.
    ``"weighted"`` (needs a positive-definite ``cov_y`` and ``objective_is_weighted=True``): C_p = (J^T C_y^-1 J)^-1, valid ONLY
        if the estimator minimised chi^2 = r^T C_y^-1 r with the same C_y; the function cannot verify that, so the caller must
        assert it. With ``residuals`` the chi^2, dof and chi^2/dof are REPORTED, never used to rescale the covariance.
    ``"residual_variance"`` (opt-in; ``cov_y`` not accepted; needs ``residuals``, N - p >= 1): C_p = s^2 (J^T J)^-1,
        s^2 = RSS/(N - p). Assumes zero-mean, equal-variance, uncorrelated errors, a correct model and a converged fit; the
        scatter of s^2 itself (about 1/sqrt(2(N-p))) is returned. It is residual-scaled, NOT a measurement uncertainty; if the
        residuals are model discrepancy it measures discrepancy. RSS = 0 is refused (a vanishing residual is not zero
        uncertainty).
    ``"exact"`` (square J, N = p, dof = 0, needs ``cov_y``): C_p = A C_y A^T with A = J^-1. No chi^2, no residual: the fit
        reproduces the data exactly, so nothing is left to check. Identical to the weighted formula for positive-definite C_y.

    Ordering contract: row i of ``jac`` is observation i; ``cov_y`` rows and columns must be in EXACTLY that order. If the
    observations were sorted or permuted for the model, pass ``measurement_order`` (a permutation: observation i corresponds to
    measurement index ``measurement_order[i]``) and the covariance is permuted as C[o][:, o]; with a ``MeasurementSet`` this is
    ``reordered(order)``. ``observation_names`` (optional) must equal the covariance's names in that order or a ValueError is
    raised; for a bare array the alignment cannot be verified and the result says so.

    Provenance: for a ``MeasurementSet`` the result carries its kind, per-element kinds and scenario qualification. Unavailable
    information is ``covariance=None`` with a reason, never zeros."""
    if mode not in FIT_COVARIANCE_MODES:
        raise ValueError(f"mode must be one of {FIT_COVARIANCE_MODES}, got {mode!r}")
    j = np.asarray(jac, dtype=float)
    if j.ndim != 2 or j.size == 0:
        raise ValueError("jac must be a non-empty 2-D array (observations x parameters)")
    if not np.all(np.isfinite(j)):
        raise ValueError("jac contains non-finite values")
    n_obs, n_par = j.shape
    names = tuple(param_names) if param_names is not None else tuple(f"p{i}" for i in range(n_par))
    if len(names) != n_par:
        raise ValueError(f"param_names has {len(names)} entries but the Jacobian has {n_par} columns")
    obs_names = None if observation_names is None else tuple(str(x) for x in observation_names)
    if obs_names is not None and len(obs_names) != n_obs:
        raise ValueError(f"observation_names has {len(obs_names)} entries but the Jacobian has {n_obs} rows")
    dof = n_obs - n_par

    # ---- scales
    if scales is not None:
        sc = np.asarray(scales, dtype=float)
        scale_note = "parameter scales S_p supplied by the caller"
    elif param_values is not None:
        pv = np.asarray(param_values, dtype=float)
        if pv.shape != (n_par,) or not np.all(np.isfinite(pv)):
            raise ValueError("param_values must be finite, one per parameter")
        sc = np.where(np.abs(pv) > 0, np.abs(pv), 1.0)
        scale_note = "parameter scales S_p = |p_hat| (1 where p_hat = 0)"
    else:
        sc, scale_note = np.ones(n_par), "parameter scales S_p = 1"
    if sc.shape != (n_par,) or np.any(sc <= 0) or not np.all(np.isfinite(sc)):
        raise ValueError("scales must be positive, finite, one per parameter")

    # ---- residuals
    r = None
    if residuals is not None:
        r = np.asarray(residuals, dtype=float)
        if r.shape != (n_obs,) or not np.all(np.isfinite(r)):
            raise ValueError(f"residuals must be a finite 1-D array of length {n_obs}")
    if mode == "residual_variance" and r is None:
        raise ValueError("mode 'residual_variance' needs the residuals (model minus data, in observation order)")
    if mode == "residual_variance" and cov_y is not None:
        raise ValueError("mode 'residual_variance' estimates the scatter from the residuals; cov_y is not accepted")
    if mode != "residual_variance" and cov_y is None:
        raise ValueError(f"mode {mode!r} needs an explicit cov_y (a MeasurementSet or covariance array); none is inferred")
    if mode == "weighted" and not objective_is_weighted:
        raise ValueError("mode 'weighted' is valid only for an estimator that minimised chi^2 = r^T C_y^-1 r with this C_y; "
                         "assert that with objective_is_weighted=True, or use mode 'sandwich' for an unweighted estimator")
    if mode == "exact" and n_obs != n_par:
        raise ValueError(f"mode 'exact' needs a square Jacobian (N = p); got {n_obs} observations and {n_par} parameters")

    # ---- covariance of the observations, aligned
    cov = None
    provenance, elems, scen, notes = "residual_scaled", None, False, []
    if cov_y is not None:
        order = None if measurement_order is None else _check_permutation(measurement_order, n_obs)
        if isinstance(cov_y, MeasurementSet):
            ms = cov_y
            if ms.n != n_obs:
                raise ValueError(f"cov_y has {ms.n} entries but the Jacobian has {n_obs} rows")
            if order is not None:
                ms = ms.reordered(order)
            if obs_names is not None and tuple(ms.names) != obs_names:
                raise ValueError("observation ordering contract violated: cov_y names (after measurement_order) "
                                 f"{tuple(ms.names)} differ from observation_names {obs_names}")
            if obs_names is None:
                notes.append("alignment of cov_y with the Jacobian rows was not verified (no observation_names given)")
            cov = np.array(ms.cov)
            provenance, elems, scen = ms.kind, ms.element_kinds, ms.is_scenario
            src_txt = ms.describe()
        else:
            cov = np.asarray(cov_y, dtype=float)
            if cov.shape != (n_obs, n_obs):
                raise ValueError(f"cov_y shape {cov.shape} does not match {n_obs} observations")
            validate_covariance(cov, name="cov_y")
            if order is not None:
                cov = cov[np.ix_(order, order)]
            provenance, src_txt = "unspecified", "covariance supplied as a bare array; its provenance is not stated"
            notes.append("bare-array covariance: alignment with the Jacobian rows cannot be verified")
        cov = 0.5 * (cov + cov.T)

    def _unavail(kind, reason, idn=None, status=None):
        return FitCovariance.unavailable(kind, reason, status=status, param_names=names, provenance=provenance, is_scenario=scen,
                                         n_obs=n_obs, n_params=n_par, dof=dof, identifiability=idn, messages=tuple(notes))

    if mode == "residual_variance":
        if dof < 1:
            return _unavail("refused", f"N - p = {dof} < 1: no degrees of freedom, so no residual variance can be estimated")
        rss0 = float(r @ r)
        if rss0 <= 0.0:
            return _unavail("refused", "RSS = 0: the residual variance estimate is zero because the data are reproduced exactly; "
                                       "this is not a statement of zero uncertainty")

    # ---- structural identifiability (forward); whitened only for the weighted mode
    wh_cov = None
    if mode == "weighted":
        ck = validate_covariance(cov, name="cov_y")
        if not ck.is_pd:
            raise CovarianceError(f"weighted fitting needs a positive-definite cov_y (rank {ck.rank} of {ck.n}); singular matrices "
                                  "are refused, not regularised. Use mode 'sandwich' (valid for singular covariances) or "
                                  "reformulate in independent variables.")
        wh_cov = cov
    idn = identifiability(j, cov_y=wh_cov, scales=sc, orientation="forward", param_names=names, flag_cond=flag_cond,
                          refuse_cond=refuse_cond, rank_rtol=rank_rtol, scales_note=scale_note)
    if idn.status == "refused":
        return _unavail("refused", "the Jacobian is not identifiable for these parameters: " + "; ".join(idn.messages), idn)

    # ---- computation (QR of the scaled Jacobian: exact left inverse of a full-column-rank matrix; no pseudoinverse)
    js = j * sc[None, :]
    chi2 = chi2_per_dof = s2 = s2_rel = None
    rss = None if r is None else float(r @ r)
    if mode == "weighted":
        low = cholesky(cov, lower=True)
        jw = solve_triangular(low, js, lower=True)
        _, rr = np.linalg.qr(jw)
        rinv = solve_triangular(rr, np.eye(n_par))
        cp = (rinv @ rinv.T) * np.outer(sc, sc)
        if r is not None:
            rw = solve_triangular(low, r, lower=True)
            chi2 = float(rw @ rw)
            chi2_per_dof = chi2 / dof if dof > 0 else None
            if dof == 0:
                notes.append("zero degrees of freedom: chi^2 carries no goodness-of-fit information")
        formula = "(J^T C_y^-1 J)^-1 for an estimator that minimised chi^2 = r^T C_y^-1 r with this C_y"
    else:
        q, rr = np.linalg.qr(js)
        rinv = solve_triangular(rr, np.eye(n_par))
        a_left = (rinv @ q.T) * sc[:, None]                                  # (J^T J)^-1 J^T (exact left inverse), p x N
        if mode == "residual_variance":
            s2 = rss / dof
            s2_rel = 1.0 / float(np.sqrt(2.0 * dof))
            cp = s2 * (rinv @ rinv.T) * np.outer(sc, sc)
            formula = f"s^2 (J^T J)^-1 with s^2 = RSS/(N-p) = {s2:.6g} (N-p = {dof}); relative scatter of s^2 about {s2_rel:.2g}"
        else:
            cp = a_left @ cov @ a_left.T
            formula = ("A C_y A^T with A = (J^T J)^-1 J^T for the UNWEIGHTED estimator" if mode == "sandwich"
                       else "A C_y A^T with A = J^-1 (square, zero degrees of freedom)")
    cp = 0.5 * (cp + cp.T)
    std = np.sqrt(np.maximum(np.diag(cp), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cp / np.outer(std, std)
    status = idn.status
    msgs = list(idn.messages) + notes
    if mode == "residual_variance":
        label = (f"RESIDUAL-SCALED covariance, {formula}. Assumes zero-mean, equal-variance, uncorrelated errors and a correct model; "
                 "it is not a measurement uncertainty and contains no separate model discrepancy; first order.")
    else:
        head = ("SENSITIVITY SCENARIO: " if scen else "") + f"first-order {mode} covariance, {formula}; input covariance: {src_txt}"
        label = head + "; conditional on the model; excludes model discrepancy; not a goodness-of-fit"
    return FitCovariance(covariance=_frozen(cp), std=_frozen(std), correlation=_frozen(corr), kind=mode, status=status, reason="",
                         provenance=provenance, is_scenario=scen, label=label, param_names=names, observation_names=obs_names,
                         n_obs=n_obs, n_params=n_par, dof=dof, rss=rss, s2=s2, s2_relative_uncertainty=s2_rel, chi2=chi2,
                         chi2_per_dof=chi2_per_dof, identifiability=idn, messages=tuple(msgs))
