r"""Mercury green line 546.1 nm (7s 3S1 -> 6p 3P2): Zeeman pattern and the pressure-scanned Fabry-Perot of the teaching lab.

Follows the lab handout "Effet Zeeman: determination experimentale du magneton de Bohr" (Hg lamp, B = 2340 G at 1 A, Fabry-Perot in
a chamber whose air pressure is varied, photomultiplier recording):
* Lande factor g = 1 + [J(J+1) + S(S+1) - L(L+1)] / [2 J (J+1)]  ->  g(3S1) = 2, g(3P2) = 3/2 (LS coupling).
* Component wavenumbers sigma = sigma0 + (mu_B B / hc) (g_u M_u - g_l M_l); strengths are Clebsch-Gordan squares of J = 1 -> J' = 2.
* Index of the gas between the mirrors: n - 1 = (n0 - 1) (273 / T) (P / 760), n0 = 1.0003 (P in mmHg, T in K).
* Fringe at the centre: order p = 2 n e sigma (e = mirror spacing). Two peaks differing in pressure by dP differ in wavenumber by
  d_sigma = sigma0 dn / n_mean, dn = (dn/dP) dP; for the two observed components d_sigma = 2 mu_B B / hc, so
  mu_B / hc = d_sigma / (2 B).
The spacing e is NOT in the handout and not assumed by the analysis functions; the simulation takes it as an argument.
Photon polarisation labels: sigma+- means M_u - M_l = +-1 (angular momentum carried by the emitted photon); the handout's sign
convention may be the opposite one. With a polariser perpendicular to B (transverse view) the sigma components are seen as linear
light perpendicular to B; the handout's green filter also acts as a polariser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MU_B_EV, ev_to_wavenumber

LAMBDA0_NM = 546.1
N_AIR_NORMAL = 1.0003
CM1_PER_TESLA = float(ev_to_wavenumber(MU_B_EV))            # mu_B / (h c) in cm^-1 per tesla (CODATA: 0.46686)


def lande_g(j: float, s: float, l: float) -> float:
    """Lande factor of an LS level: 1 + [J(J+1) + S(S+1) - L(L+1)] / [2 J (J+1)]."""
    return 1 + (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))


G_UPPER = lande_g(1, 1, 0)       # 7s 3S1 -> 2
G_LOWER = lande_g(2, 1, 1)       # 6p 3P2 -> 3/2

# squared Clebsch-Gordan coefficients <2 M_l | 1 M_u ; 1 q>^2 (relative line strengths), keyed by (M_u, M_l)
_STRENGTH = {(1, 2): 1.0, (1, 1): 1 / 2, (1, 0): 1 / 6, (0, 1): 1 / 2, (0, 0): 2 / 3, (0, -1): 1 / 2,
             (-1, 0): 1 / 6, (-1, -1): 1 / 2, (-1, -2): 1.0}


def components(b_tesla: float = 0.234) -> pd.DataFrame:
    """The nine Zeeman components: M_u, M_l, polarisation (sigma+, sigma-, pi), delta = g_u M_u - g_l M_l, relative strength,
    wavenumber shift (cm^-1) and wavelength shift (pm). delta runs from -2 to +2 in steps of 1/2."""
    rows = []
    for (mu, ml), s in _STRENGTH.items():
        delta = G_UPPER * mu - G_LOWER * ml
        pol = {1: "sigma+", -1: "sigma-", 0: "pi"}[mu - ml]
        dsig = delta * CM1_PER_TESLA * b_tesla
        dlam_pm = -dsig * (LAMBDA0_NM * 1e-7) ** 2 * 1e10          # d_lambda = -lambda^2 d_sigma, cm -> pm
        rows.append((mu, ml, pol, delta, s, dsig, dlam_pm))
    df = pd.DataFrame(rows, columns=["m_upper", "m_lower", "polarisation", "delta", "strength", "dsigma_cm1", "dlambda_pm"])
    return df.sort_values("delta").reset_index(drop=True)


OBSERVED = ((1, 2), (-1, -2))                              # the two components the handout analyses


def dn_dp(temperature_k: float, n0: float = N_AIR_NORMAL) -> float:
    """Index change per mmHg: (n0 - 1) 273 / (760 T)."""
    return (n0 - 1) * 273.0 / (760.0 * temperature_k)


def index(p_mmhg, temperature_k: float, n0: float = N_AIR_NORMAL):
    """n(P, T) = 1 + (n0 - 1) (273 / T) (P / 760)."""
    return 1 + (n0 - 1) * (273.0 / temperature_k) * (np.asarray(p_mmhg, dtype=float) / 760.0)


def bohr_from_peak_spacing(dp_mmhg: float, b_gauss: float, temperature_k: float, p_mean_mmhg: float = 450.0,
                           wavelength_nm: float = LAMBDA0_NM) -> float:
    """The handout's analysis: mu_B / (hc) in cm^-1 per gauss from the pressure difference between the two neighbouring peaks
    belonging to the two observed components. d_sigma = sigma0 dn / n_mean and mu_B/hc = d_sigma / (2 B)."""
    dn = dn_dp(temperature_k) * dp_mmhg
    n_mean = float(index(p_mean_mmhg, temperature_k))
    dsigma = (1e7 / wavelength_nm) * dn / n_mean                    # cm^-1
    return dsigma / (2 * b_gauss)


def bohr_uncertainty(dp_mmhg: float, sigma_dp: float, b_gauss: float, sigma_b: float, temperature_k: float,
                     **kw) -> float:
    """Standard uncertainty of mu_B/hc from the reading error of dP and the field (quadrature of the two relative errors;
    the temperature and the index of air enter at the 1e-3 level and are ignored here)."""
    v = bohr_from_peak_spacing(dp_mmhg, b_gauss, temperature_k, **kw)
    return v * float(np.hypot(sigma_dp / dp_mmhg, sigma_b / b_gauss))


def predicted_peak_spacing_mmhg(b_gauss: float, temperature_k: float, wavelength_nm: float = LAMBDA0_NM,
                                p_mean_mmhg: float = 450.0) -> float:
    """What the handout predicts for dP between the two observed peaks: 2 (mu_B/hc) B n_mean lambda / (dn/dP)."""
    dsigma = 2 * CM1_PER_TESLA * 1e-4 * b_gauss                     # cm^-1
    dn = dsigma * wavelength_nm * 1e-7 * float(index(p_mean_mmhg, temperature_k))
    return dn / dn_dp(temperature_k)


def pressure_scan(p_mmhg, b_tesla: float, e_mm: float, temperature_k: float = 293.0, reflectivity: float = 0.9,
                  observed_only: bool = False, polarisation: str = "sigma", wavelength_nm: float = LAMBDA0_NM):
    """Photomultiplier signal at the centre of the rings against the air pressure in the Fabry-Perot.

    Each component k contributes strength_k / (1 + F sin^2(pi p_k)) with order p_k = 2 n e sigma_k, F = 4R/(1-R)^2.
    polarisation 'sigma' keeps the six sigma components (polariser perpendicular to B), 'pi' the three pi ones, 'all' all nine;
    observed_only keeps only the two components analysed in the handout."""
    df = components(b_tesla)
    if observed_only:
        df = df[[(a, b) in OBSERVED for a, b in zip(df.m_upper, df.m_lower)]]
    elif polarisation == "sigma":
        df = df[df.polarisation != "pi"]
    elif polarisation == "pi":
        df = df[df.polarisation == "pi"]
    n = index(p_mmhg, temperature_k)
    sigma0 = 1e7 / wavelength_nm                                   # cm^-1
    f = 4 * reflectivity / (1 - reflectivity) ** 2
    out = np.zeros_like(n)
    for _, r in df.iterrows():
        order = 2 * n * (e_mm * 0.1) * (sigma0 + r.dsigma_cm1)     # e in cm
        out += r.strength / (1 + f * np.sin(np.pi * order) ** 2)
    return out


def fig_mercury_scan(e_mm: float = 10.0, temperature_k: float = 293.0, b_gauss: float = 2340.0):
    """Simulated recording without and with the field (the handout's B), and the component diagram. e_mm is an ILLUSTRATIVE spacing."""
    import matplotlib.pyplot as plt
    p = np.linspace(100, 760, 20000)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    y0 = pressure_scan(p, 0.0, e_mm, temperature_k, polarisation="all")
    ax[0].plot(p, y0 / y0.max(), color="tab:green")
    ax[0].set_title("without B", fontsize=10)
    y1 = pressure_scan(p, b_gauss * 1e-4, e_mm, temperature_k)
    ax[1].plot(p, y1 / y1.max(), color="tab:green")
    ax[1].set_title(f"with B = {b_gauss:g} G (six sigma components)", fontsize=10)
    for a in ax[:2]:
        a.set_xlabel("pressure in the Fabry-Perot (mmHg)")
        a.set_ylabel("signal (arb.)")
        a.invert_xaxis()
    df = components(b_gauss * 1e-4)
    col = {"sigma+": "tab:red", "sigma-": "tab:blue", "pi": "tab:green"}
    for _, r in df.iterrows():
        ax[2].vlines(r.delta, 0, r.strength, color=col[r.polarisation], lw=4)
    ax[2].set_xlabel(r"$g_uM_u - g_lM_l$")
    ax[2].set_ylabel("relative strength")
    ax[2].set_title(r"546.1 nm: $7s\,^3S_1 \to 6p\,^3P_2$", fontsize=10)
    fig.suptitle(f"Virtual lab (spacing e = {e_mm:g} mm is an illustrative assumption)", fontsize=10)
    fig.tight_layout()
    return fig


def analyse_recording(dp_order_mmhg: float, dp_pair_mmhg: float, b_gauss: float, temperature_k: float,
                      p_mean_mmhg: float = 450.0, sigma_dp_mmhg: float = 0.0, sigma_b_gauss: float = 0.0, wavelength_nm: float = LAMBDA0_NM) -> dict:
    """The lab analysis in one call.

    dp_order_mmhg: pressure between two successive peaks of the same component (one order, 'peak to peak');
    dp_pair_mmhg: pressure between the two neighbouring peaks of the two observed components (the doublet spacing);
    p_mean_mmhg: the mean pressure Pm of those two peaks, which sets the mean index n_m of the handout's formula.
    Returns the mirror spacing e (from the order: dn = lambda / (2 e)), mu_B/hc in cm^-1/G, its uncertainty, the CODATA value and
    the relative deviation. e is only a by-product: the handout's method does not need it."""
    e_mm = wavelength_nm * 1e-6 / (2 * dn_dp(temperature_k) * dp_order_mmhg)         # nm -> mm
    mu = bohr_from_peak_spacing(dp_pair_mmhg, b_gauss, temperature_k, p_mean_mmhg=p_mean_mmhg, wavelength_nm=wavelength_nm)
    u = bohr_uncertainty(dp_pair_mmhg, sigma_dp_mmhg, b_gauss, sigma_b_gauss, temperature_k, p_mean_mmhg=p_mean_mmhg,
                         wavelength_nm=wavelength_nm)
    ref = CM1_PER_TESLA * 1e-4
    return {"e_mm": e_mm, "mu_b_hc_cm1_per_gauss": mu, "uncertainty": u, "codata": ref, "relative_deviation": mu / ref - 1,
            "pair_over_order": dp_pair_mmhg / dp_order_mmhg}
