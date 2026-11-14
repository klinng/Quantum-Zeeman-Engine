r"""Two valence electrons (alkaline-earth-like atoms, e.g. calcium 4s4p): singlet and triplet P terms in a magnetic field.

One electron sits in an s orbital (l = 0) and one in a p orbital (l = 1). Basis: |m_l> x |m_s1> x |m_s2>, 12 states.

    H = E_t + DELTA (1/4 - s1.s2) + zeta l.s2 + mu_B B (L_z + g_s (S1z + S2z))

* DELTA (1/4 - s1.s2) is the exchange splitting: zero for the triplet (S = 1), DELTA for the singlet (S = 0).
* zeta l.s2 is the spin-orbit coupling of the p electron. Inside the triplet it gives the Lande interval rule
  (E(3P2) - E(3P1)) : (E(3P1) - E(3P0)) = 2 : 1, and it is NOT diagonal in S: it mixes 1P1 into 3P1. That mixing
  makes the "forbidden" intercombination line 1S0 - 3P1 (657 nm in calcium) weakly allowed.
* The Zeeman term is the same as for one electron, with the total spin S = s1 + s2.
Not modelled: spin-spin interaction (large in helium, which is why this model suits calcium rather than helium),
configuration mixing with other ODD-parity configurations that carry J = 1 (4s5p and 3d4p; see CA_ODD_J1_LEVELS_CM1; 4s3d is
even and cannot mix with 4s4p), and hyperfine structure. Energies in eV, fields in tesla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .constants import G_S, MU_B_EV, ev_to_wavenumber, wavenumber_to_ev
from .hamiltonians import angular_momentum_ops

_LX, _LY, _LZ = angular_momentum_ops(1)
_SX, _SY, _SZ = angular_momentum_ops(0.5)
_I3, _I2 = np.eye(3), np.eye(2)


def _op(l=None, s1=None, s2=None) -> np.ndarray:
    """Operator on the 12-dimensional space from factors acting on (orbital, spin 1, spin 2)."""
    return np.kron(np.kron(_I3 if l is None else l, _I2 if s1 is None else s1), _I2 if s2 is None else s2)


def operators() -> dict[str, np.ndarray]:
    o = {}
    for c, lc, sc_ in zip("xyz", (_LX, _LY, _LZ), (_SX, _SY, _SZ)):
        o["L" + c] = _op(l=lc)
        o["s1" + c] = _op(s1=sc_)
        o["s2" + c] = _op(s2=sc_)
        o["S" + c] = o["s1" + c] + o["s2" + c]
    return o


def h_two_electron(b_tesla: float, zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0, g_s: float = G_S) -> np.ndarray:
    """12x12 Hamiltonian (eV). e_t_ev is the energy of the triplet term before spin-orbit coupling."""
    o = operators()
    s1s2 = sum(o["s1" + c] @ o["s2" + c] for c in "xyz")
    ls2 = sum(o["L" + c] @ o["s2" + c] for c in "xyz")
    h = e_t_ev * np.eye(12) + delta_ev * (0.25 * np.eye(12) - s1s2) + zeta_ev * ls2
    return h + MU_B_EV * b_tesla * (o["Lz"] + g_s * o["Sz"])


def classify_states(b_tesla: float, zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0, g_s: float = G_S) -> pd.DataFrame:
    """Eigenstates with their labels. A tiny extra J_z term is added at B = 0 only to pick definite M_J inside degenerate levels.

    Columns: energy_ev, mj (<J_z>), m_j (exact integer), j (the DOMINANT J: the J = 0, 1, 2 subspace with the largest
    weight), j_purity (that weight), j2 (<J^2>), p_j0, p_j1, p_j2 (weights), singlet (probability of S = 0), label
    ('3P0','3P1','3P2','1P1'). J is a good quantum number only at B = 0; at finite field use j_purity (for calcium it
    is above 1 - 1e-3 up to a few tesla because the fine structure is 50-100 cm^-1 against a Zeeman energy of 0.47 cm^-1
    per tesla). The label of a level is its dominant character: with 1P1 and 3P1 mixed by the spin-orbit term, the J = 1
    level that is mostly singlet is called 1P1 and the other 3P1.
    """
    o = operators()
    jz = o["Lz"] + o["Sz"]
    j2 = sum((o["L" + c] + o["S" + c]) @ (o["L" + c] + o["S" + c]) for c in "xyz")
    wj, uj = np.linalg.eigh(j2)
    proj = {jj: uj[:, np.abs(wj - jj * (jj + 1)) < 1e-8] @ uj[:, np.abs(wj - jj * (jj + 1)) < 1e-8].conj().T for jj in (0, 1, 2)}
    p_singlet = 0.25 * np.eye(12) - sum(o["s1" + c] @ o["s2" + c] for c in "xyz")
    h = h_two_electron(b_tesla, zeta_ev, delta_ev, e_t_ev, g_s)
    if b_tesla == 0.0:
        h = h + 1e-10 * jz
    w, v = np.linalg.eigh(h)
    ex = lambda m, k: float(np.real(v[:, k].conj() @ m @ v[:, k]))
    rows = []
    for k in range(12):
        pj = [ex(proj[jj], k) for jj in (0, 1, 2)]
        jmax = int(np.argmax(pj))
        mj = ex(jz, k)
        rows.append((w[k], mj, int(round(mj)), jmax, pj[jmax], ex(j2, k), *pj, ex(p_singlet, k)))
    df = pd.DataFrame(rows, columns=["energy_ev", "mj", "m_j", "j", "j_purity", "j2", "p_j0", "p_j1", "p_j2", "singlet"])
    df["label"] = np.where(df.j == 0, "3P0", np.where(df.j == 2, "3P2", np.where(df.singlet > 0.5, "1P1", "3P1")))
    return df


def h_two_electron_general(b_tesla: float, zeta_s_ev: float, zeta_a_ev: float, delta_ev: float, e_t_ev: float = 0.0,
                           g_s: float = G_S) -> np.ndarray:
    """Same 12 states, with the spin-orbit term split into its symmetric and antisymmetric parts:
    zeta_S L.S + zeta_A L.(s2 - s1), L = l of the p electron. Because l.s2 = (1/2) L.S + (1/2) L.(s2 - s1), the
    one-parameter model h_two_electron is the special case zeta_S = zeta_A = zeta / 2. Only zeta_A connects S = 1 with S = 0."""
    o = operators()
    s1s2 = sum(o["s1" + c] @ o["s2" + c] for c in "xyz")
    ls = sum(o["L" + c] @ o["S" + c] for c in "xyz")
    lda = sum(o["L" + c] @ (o["s2" + c] - o["s1" + c]) for c in "xyz")
    h = e_t_ev * np.eye(12) + delta_ev * (0.25 * np.eye(12) - s1s2) + zeta_s_ev * ls + zeta_a_ev * lda
    return h + MU_B_EV * b_tesla * (o["Lz"] + g_s * o["Sz"])


def calcium_four_parameter_fit(e3p0_cm1: float, e3p1_cm1: float, e3p2_cm1: float, e1p1_cm1: float) -> dict[str, float]:
    """Exact (closed-form) solution for (zeta_S, zeta_A, Delta, E_t) in cm^-1 from the four measured 4s4p levels.

    This is a CALIBRATION: four parameters reproduce four numbers, so nothing is left to check. 3P0 and 3P2 do not involve
    zeta_A or the singlet at all, so E(3P2) - E(3P0) = 3 zeta_S and E_t = E(3P2) - zeta_S. The 3P1 / 1P1 pair is a 2 x 2 block
    [[a, V], [V, b]] with a = E_t - zeta_S (the interval-rule position of 3P1), b = E_t + Delta, V = sqrt(2) zeta_A. Its eigenvalues
    are the measured 3P1 (l1) and 1P1 (l2), so V^2 = (a - l1)(l2 - a) and the singlet admixture of 3P1 is exactly
    eps^2 = (a - l1) / (l2 - l1): the displacement of 3P1 from the interval rule, divided by the singlet-triplet separation.
    Only V^2 enters the energies, so the sign of zeta_A is not determined; zeta_A >= 0 is returned (observable-neutral).
    Raises ValueError if the data give V^2 < 0 (3P1 below the interval-rule position would need an imaginary coupling)."""
    zeta_s = (e3p2_cm1 - e3p0_cm1) / 3.0
    e_t = e3p2_cm1 - zeta_s
    a = e_t - zeta_s
    v2 = (a - e3p1_cm1) * (e1p1_cm1 - a)
    if v2 < 0:
        raise ValueError("no real zeta_A reproduces these levels (3P1 above its interval-rule position)")
    b = e3p1_cm1 + e1p1_cm1 - a
    return {"zeta_S": zeta_s, "zeta_A": float(np.sqrt(v2 / 2.0)), "Delta": b - e_t, "E_t": e_t,
            "eps2": (a - e3p1_cm1) / (e1p1_cm1 - e3p1_cm1), "displacement_3P1": a - e3p1_cm1}


def calcium_fit_sensitivity(levels_cm1: dict[str, float], sigma_cm1: float | dict[str, float]) -> pd.DataFrame:
    """Linear propagation of ASSUMED level uncertainties through calcium_four_parameter_fit.

    The NIST Handbook levels (source code SC85) come with no uncertainties, so sigma_cm1 is a scenario, not a data value:
    a single number for all four levels, or a dict keyed by '3P0', '3P1', '3P2', '1P1'. Returns, for each output
    (zeta_S, zeta_A, Delta, E_t, eps2), the value and ``scenario_std``: the standard deviation PROPAGATED from the assumed
    level uncertainties (sqrt of the diagonal of J C J^T, C = diag(sigma^2), independent levels). It is a sensitivity
    scenario, NOT an experimental standard error and NOT a fit-residual error (the fit has zero degrees of freedom).
    The correlation matrix is attached as df.attrs['correlation']. The Jacobian is by central differences.

    Compatibility: the column ``std_err`` (the former name) is kept as an exact copy of ``scenario_std`` and is deprecated;
    new code should use ``scenario_std``."""
    keys = ["3P0", "3P1", "3P2", "1P1"]
    sig = np.array([sigma_cm1[k] if isinstance(sigma_cm1, dict) else sigma_cm1 for k in keys], dtype=float)
    out = ["zeta_S", "zeta_A", "Delta", "E_t", "eps2"]
    f = lambda x: np.array([calcium_four_parameter_fit(*x)[o] for o in out])
    x0 = np.array([levels_cm1[k] for k in keys], dtype=float)
    jac = np.empty((len(out), 4))
    for i in range(4):
        h = 1e-4
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        jac[:, i] = (f(xp) - f(xm)) / (2 * h)
    cov = jac @ np.diag(sig ** 2) @ jac.T
    err = np.sqrt(np.diag(cov))
    df = pd.DataFrame({"value": f(x0), "scenario_std": err, "std_err": err}, index=out)     # std_err: deprecated alias
    df.attrs["note"] = "scenario_std is propagated from ASSUMED level uncertainties; std_err is a deprecated alias of it"
    df.attrs["correlation"] = pd.DataFrame(cov / np.outer(err, err), index=out, columns=out)
    return df


def zero_field_levels_cm1(zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0) -> dict[str, float]:
    """Energies (cm^-1) of 3P0, 3P1, 3P2 and 1P1 from diagonalisation at zero field."""
    df = classify_states(0.0, zeta_ev, delta_ev, e_t_ev)
    return {lab: float(ev_to_wavenumber(g.energy_ev.mean())) for lab, g in df.groupby("label")}


def fit_to_levels(e3p0_cm1: float, e3p1_cm1: float, e3p2_cm1: float, e1p1_cm1: float):
    """Fit (zeta, Delta, E_t) so the model reproduces the four term energies (3 parameters, 4 numbers: one consistency check).
    Returns (zeta_eV, delta_eV, e_t_eV, residuals_cm1 model minus input)."""
    target = np.array([e3p0_cm1, e3p1_cm1, e3p2_cm1, e1p1_cm1])

    def resid(p):
        lv = zero_field_levels_cm1(wavenumber_to_ev(p[0]), wavenumber_to_ev(p[1]), wavenumber_to_ev(p[2]))
        return np.array([lv["3P0"], lv["3P1"], lv["3P2"], lv["1P1"]]) - target

    p0 = [e3p2_cm1 - e3p1_cm1, e1p1_cm1 - e3p1_cm1, e3p1_cm1]
    sol = least_squares(resid, p0, x_scale=[100, 1000, 10000], xtol=1e-14, ftol=1e-14)
    z, d, et = (wavenumber_to_ev(x) for x in sol.x)
    return z, d, et, resid(sol.x)


def singlet_admixture_in_3p1(zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0) -> float:
    """Probability of S = 0 character in the 3P1 level (the squared mixing amplitude epsilon^2)."""
    df = classify_states(0.0, zeta_ev, delta_ev, e_t_ev)
    return float(df[df.label == "3P1"].singlet.mean())


def g_factor(label: str, zeta_ev: float, delta_ev: float, e_t_ev: float = 0.0, b_tesla: float = 1e-4) -> float:
    """g_J of a level from the model's own eigenvalues at a tiny field: (E(M_J = +J) - E(M_J = -J)) / (2 J mu_B B)."""
    if label == "3P0":
        raise ValueError("3P0 (J = 0) has no Zeeman splitting")
    df = classify_states(b_tesla, zeta_ev, delta_ev, e_t_ev)
    g = df[df.label == label].sort_values("energy_ev")
    j = float(g.j.iloc[0])
    return float((g.energy_ev.iloc[-1] - g.energy_ev.iloc[0]) / (2 * j * MU_B_EV * b_tesla))


def intercombination_rate(eps2: float, a_singlet_per_s: float, lambda_singlet_nm: float, lambda_triplet_nm: float) -> float:
    """Rough decay rate (1/s) of 3P1 to the ground state: the singlet admixture eps^2 times the 1P1 rate, scaled by
    (lambda_1P1 / lambda_3P1)^3 because spontaneous emission goes as the cube of the frequency. Assumes the singlet and the
    admixed triplet share the same dipole matrix element: an order-of-magnitude estimate, not a precise prediction."""
    return eps2 * a_singlet_per_s * (lambda_singlet_nm / lambda_triplet_nm) ** 3


# Calcium 4s4p term energies (cm^-1).
# 3P1: the measured 1S0-3P1 frequency 455 986 240 494 135.8 (3.4) Hz (tf.nist.gov/general/pdf/2197.pdf) divided by c
# gives 15210.0638 cm^-1 (the NIST Handbook table lists 15210.063).
# 3P0, 3P2, 1P1: NIST Handbook of Basic Atomic Spectroscopic Data, "Energy Levels of Neutral Calcium (Ca I)", whose
# source code is SC85 (Sugar and Corliss 1985): 15157.901, 15315.943 and 23652.304 cm^-1.
CA_3P1_MEASURED_HZ = 455_986_240_494_135.8
CA_LEVELS_CM1 = {"3P0": 15157.901, "3P1": 15210.0638, "3P2": 15315.943, "1P1": 23652.304}

# Odd-parity J = 1 levels of Ca I that can mix with 4s4p 3P1 and 1P1 (same parity, same J): NIST Handbook, "Energy Levels of
# Neutral Calcium (Ca I)", source code SC85, no uncertainties listed. 4s3d (even parity) cannot mix with 4s4p, and the
# J = 2 levels 3d4p 3F2 (35730.454) and 1D2 (35835.413) cannot mix with J = 1 in a rotationally invariant Hamiltonian.
CA_ODD_J1_LEVELS_CM1 = {"4s4p 3P1": 15210.0638, "4s4p 1P1": 23652.304, "4s5p 3P1": 36554.749, "4s5p 1P1": 36731.615,
                        "3d4p 3D1": 38192.392}


def odd_j1_perturber_table() -> pd.DataFrame:
    """Energy separations (cm^-1) of the other odd-parity J = 1 levels from 4s4p 3P1 and 1P1. A perturber at distance D
    shifts a level by V^2 / D and admixes (V / D)^2 of itself, so the farther ones need much larger couplings V to matter;
    the couplings themselves are NOT known here (they need a configuration-interaction calculation or measured data)."""
    lv = CA_ODD_J1_LEVELS_CM1
    rows = [(k, v, v - lv["4s4p 3P1"], v - lv["4s4p 1P1"]) for k, v in lv.items() if k not in ("4s4p 3P1", "4s4p 1P1")]
    return pd.DataFrame(rows, columns=["level", "energy_cm1", "above_3P1_cm1", "above_1P1_cm1"])


def intercombination_prediction(levels_cm1: dict[str, float], a_1p1_per_s: float = 2.20e8) -> dict[str, float]:
    """Consistency check of the 4s^2 1S0 - 4s4p 3P1 decay rate from the four-parameter ENERGY fit alone; the measured
    lifetime is not used. rate = eps^2 * A(1P1) * (sigma_3P1 / sigma_1P1)^3 (vacuum wavenumbers; emission goes as the cube of
    the frequency), eps^2 = (a - l1) / (l2 - l1). Assumes one shared dipole matrix element, so it is an order-of-magnitude
    test of the model, not a precise prediction."""
    fit = calcium_four_parameter_fit(levels_cm1["3P0"], levels_cm1["3P1"], levels_cm1["3P2"], levels_cm1["1P1"])
    scale = a_1p1_per_s * (levels_cm1["3P1"] / levels_cm1["1P1"]) ** 3
    rate = fit["eps2"] * scale
    return {"eps2": fit["eps2"], "rate_per_s": rate, "lifetime_ms": 1e3 / rate, "rate_scale_per_s": scale}


def displacement_for_rate(rate_per_s: float, levels_cm1: dict[str, float], a_1p1_per_s: float = 2.20e8) -> float:
    """The displacement of 3P1 from its interval-rule position (cm^-1) that the 1P1 admixture ALONE would need in order to
    give a measured intercombination rate: (rate / scale) * (l2 - l1). Compare with the actual displacement in the data;
    the difference is what other mechanisms (other perturbers, spin-spin) must supply."""
    scale = a_1p1_per_s * (levels_cm1["3P1"] / levels_cm1["1P1"]) ** 3
    return rate_per_s / scale * (levels_cm1["1P1"] - levels_cm1["3P1"])
