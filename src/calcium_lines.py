r"""Zeeman spectrum of the calcium 4s^2 1S0 -> 4s4p lines (the line spectrum of the two-electron model).

The ground state 4s^2 1S0 has J = 0, so it has no Zeeman shift: all structure comes from the excited 4s4p levels.
The electric-dipole operator changes one s electron into a p electron and does not touch the spin, so acting on the
spin-singlet ground state it produces the spin-singlet state with the p electron in m_l = q (q = -1, 0, +1 for the
polarisations sigma-, pi, sigma+). The line strength to an eigenstate k of the 12-state Hamiltonian is therefore

    S_k(q) = | <k | m_l = q, singlet> |^2 ,       sum over k of S_k(q) = 1  for every q.

Consequences, all visible in the output:
* at B = 0 only the J = 1 levels radiate: 1P1 carries 1 - eps^2 and the "forbidden" 3P1 carries eps^2 (the singlet admixture);
* 3P0 and 3P2 are dark at B = 0, but a magnetic field mixes 3P0 with 3P1 (same M = 0), so 3P0 gains strength
  that grows as B^2 (the field-induced clock transition of alkaline-earth atoms).
Only relative strengths are computed; absolute rates need the 1P1 decay rate (see two_electron.intercombination_rate).
The 1P1 lines use the model's own mixing; configuration mixing with the odd-parity 4s5p and 3d4p levels is not included.
Energies are measured from the ground state, whose energy is zero by construction of the level fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ev_to_wavenumber
from .two_electron import classify_states, h_two_electron

_SINGLET = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2)       # (up down - down up)/sqrt(2) in the (s1, s2) product basis


def dipole_state(q: int) -> np.ndarray:
    """The state d_q |4s^2 1S0>: orbital m_l = q, spin singlet (a 12-component vector; m_l order +1, 0, -1)."""
    orb = np.zeros(3)
    orb[1 - q] = 1.0
    return np.kron(orb, _SINGLET)


def calcium_lines(b_tesla: float, zeta_ev: float, delta_ev: float, e_t_ev: float) -> pd.DataFrame:
    """Every line of the 4s^2 1S0 -> 4s4p manifold at field B, with strength above `min_strength` kept by the caller.

    Columns: label (dominant term), wavenumber_cm1, wavelength_nm (vacuum), q (-1, 0, +1 = sigma-, pi, sigma+),
    strength (S_k(q))."""
    h = h_two_electron(b_tesla, zeta_ev, delta_ev, e_t_ev)
    w, v = np.linalg.eigh(h)
    df = classify_states(b_tesla, zeta_ev, delta_ev, e_t_ev)         # sorted by energy, same order as eigh
    rows = []
    for q in (-1, 0, 1):
        amp = v.conj().T @ dipole_state(q)
        for k in range(12):
            rows.append((df.label[k], float(ev_to_wavenumber(w[k])), q, float(abs(amp[k]) ** 2)))
    out = pd.DataFrame(rows, columns=["label", "wavenumber_cm1", "q", "strength"])
    out["wavelength_nm"] = 1e7 / out.wavenumber_cm1
    return out


def term_strength(b_tesla: float, label: str, zeta_ev: float, delta_ev: float, e_t_ev: float) -> float:
    """Total relative strength (sum over the Zeeman components and polarisations) of the lines of one term."""
    df = calcium_lines(b_tesla, zeta_ev, delta_ev, e_t_ev)
    return float(df[df.label == label].strength.sum())


def fig_calcium_lines(b_tesla: float = 1.0):
    """Left: the 3P1 (657 nm) group at B, strengths on a log scale, coloured by polarisation.
    Right: strength of the 3P0 line against B on log axes; the slope 2 is the B^2 law."""
    import matplotlib.pyplot as plt
    from .two_electron import CA_LEVELS_CM1, fit_to_levels
    z, d, et, _ = fit_to_levels(*(CA_LEVELS_CM1[k] for k in ("3P0", "3P1", "3P2", "1P1")))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    df = calcium_lines(b_tesla, z, d, et)
    g = df[(df.label == "3P1") & (df.strength > 0)]
    centre = g.wavenumber_cm1.mean()
    colours = {-1: "tab:red", 0: "tab:blue", 1: "tab:green"}
    names = {-1: r"$\sigma^-$", 0: r"$\pi$", 1: r"$\sigma^+$"}
    for q in (-1, 0, 1):
        s = g[g.q == q]
        ax1.vlines((s.wavenumber_cm1 - centre) * 29.9792458, 1e-9, s.strength, color=colours[q], lw=3, label=names[q])
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, 1e-3)
    ax1.set_xlabel("frequency from the zero-field 3P1 line (GHz)")
    ax1.set_ylabel("relative strength (1P1 line = 1)")
    ax1.set_title(f"4s$^2$ $^1$S$_0$ - 4s4p $^3$P$_1$ (657 nm) at B = {b_tesla:g} T", fontsize=10)
    ax1.legend()
    bs = np.logspace(-3, 0.5, 30)
    s0 = [term_strength(b, "3P0", z, d, et) for b in bs]
    ax2.loglog(bs, s0, "o-", ms=3, label="3P0 (model)")
    ax2.loglog(bs, s0[0] * (bs / bs[0]) ** 2, "k--", lw=1, label=r"$\propto B^2$")
    ax2.set_xlabel("magnetic field B (T)")
    ax2.set_ylabel("relative strength of the 3P0 line")
    ax2.set_title("Field-induced strength of the dark 3P0 line", fontsize=10)
    ax2.legend()
    fig.tight_layout()
    return fig
