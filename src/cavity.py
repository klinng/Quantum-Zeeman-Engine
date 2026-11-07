r"""Photon operators and the Jaynes-Cummings model: one two-level atom coupled to one cavity mode.

    H = w_c a^+ a + (w_a / 2) sigma_z + g (a sigma_+ + a^+ sigma_-)        (hbar = 1, rotating-wave approximation)

* a, a^+ act in a Fock space truncated at n_max photons. [a, a^+] = 1 holds exactly except on the last state
  (a truncation artefact), so keep n_max well above the photon number you use.
* The excitation number N = a^+ a + sigma_+ sigma_- commutes with H: the problem splits into 2x2 blocks.
* Resonance (w_c = w_a): the levels of the N-excitation block are N w_c +- g sqrt(N)  (vacuum Rabi splitting 2g).
* Starting in |excited, 0 photons>, the excited-state probability is cos^2(g t): vacuum Rabi oscillation.

Units are arbitrary (all energies in the same unit as g). This is a stand-alone demonstration; it is not tied
to a real atom or cavity, and no experimental numbers are used.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def annihilation(n_max: int) -> np.ndarray:
    """a in the Fock basis |0>..|n_max>: a|n> = sqrt(n)|n-1>."""
    return np.diag(np.sqrt(np.arange(1, n_max + 1)), k=1).astype(complex)


def creation(n_max: int) -> np.ndarray:
    return annihilation(n_max).conj().T


def jaynes_cummings(n_max: int, w_c: float, w_a: float, g: float) -> np.ndarray:
    """H in the basis (photon number) (x) (atom: index 0 = excited, 1 = ground)."""
    a, ad = annihilation(n_max), creation(n_max)
    n_op = ad @ a
    sp = np.array([[0, 1], [0, 0]], dtype=complex)          # sigma_+ : ground -> excited
    sm = sp.T.copy()
    sz = np.diag([1.0, -1.0]).astype(complex)
    i_f, i_a = np.eye(n_max + 1), np.eye(2)
    return w_c * np.kron(n_op, i_a) + 0.5 * w_a * np.kron(i_f, sz) + g * (np.kron(a, sp) + np.kron(ad, sm))


def excitation_number(n_max: int) -> np.ndarray:
    ad, a = creation(n_max), annihilation(n_max)
    sp = np.array([[0, 1], [0, 0]], dtype=complex)
    return np.kron(ad @ a, np.eye(2)) + np.kron(np.eye(n_max + 1), sp @ sp.T)


def excited_probability(n_max: int, w_c: float, w_a: float, g: float, times, n_photons: int = 0) -> np.ndarray:
    """P(atom excited) against time, starting from |n_photons photons, atom excited>."""
    h = jaynes_cummings(n_max, w_c, w_a, g)
    psi0 = np.zeros(2 * (n_max + 1), dtype=complex)
    psi0[2 * n_photons] = 1.0                                # photon number n, atom excited
    proj = np.kron(np.eye(n_max + 1), np.diag([1.0, 0.0]))
    w, v = np.linalg.eigh(h)
    c0 = v.conj().T @ psi0
    out = []
    for t in np.atleast_1d(times):
        psi = v @ (np.exp(-1j * w * t) * c0)
        out.append(float(np.real(psi.conj() @ proj @ psi)))
    return np.array(out)


def fig_jaynes_cummings(g: float = 1.0, n_max: int = 12):
    """Left: vacuum Rabi oscillation (resonant and detuned). Right: levels of the 1-excitation block against detuning
    (an avoided crossing of size 2g)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    t = np.linspace(0, 4 * np.pi / g, 400)
    for det, c in ((0.0, "tab:blue"), (2.0, "tab:orange"), (5.0, "tab:green")):
        a1.plot(t * g, excited_probability(n_max, 10.0, 10.0 + det, g, t), color=c, label=f"detuning {det:g} g")
    a1.plot(t * g, np.cos(g * t) ** 2, "k:", label="cos²(g t)")
    a1.set_xlabel("g t"); a1.set_ylabel("probability atom is excited"); a1.legend(fontsize=8)
    a1.set_title("Vacuum Rabi oscillation: atom starts excited, cavity empty", fontsize=10)
    d = np.linspace(-6, 6, 200)
    for sign in (0, 1):
        a2.plot(d, [np.linalg.eigvalsh(jaynes_cummings(n_max, 10.0, 10.0 + x, g)[np.ix_([0, 3], [0, 3])])[sign] - 5.0 for x in d], lw=2)
    a2.set_xlabel("atom minus cavity frequency (units of g)"); a2.set_ylabel("one-excitation levels − w_c/2 (units of g)")
    a2.set_title("Avoided crossing: splitting 2g at resonance", fontsize=10)
    fig.tight_layout()
    return fig
