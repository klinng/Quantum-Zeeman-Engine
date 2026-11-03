"""Create the four lab notebooks. Run once, then execute them with scripts/execute_notebooks.py
(or open them in VSCode / Jupyter and click Run All)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(exist_ok=True)

HEAD = r"""import sys
sys.path.insert(0, '..')          # so `import src...` works from the notebooks folder
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})
"""


def nb(cells):
    def cell(kind, src):
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        return c
    return {"cells": [cell(k, s) for k, s in cells], "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}


# --------------------------------------------------------------------------------------------
NB1 = [
("markdown", r"""# Level 1: The normal Zeeman effect

**Goal.** See, in numbers and plots, that a magnetic field splits an orbital level into equally
spaced components, and that this produces the famous Lorentz *triplet*.

**Physics.** An electron with orbital angular momentum $\mathbf L$ has a magnetic moment
$-\mu_B\mathbf L/\hbar$. In a field $B$ along $z$ the energy shifts by

$$\Delta E = \mu_B B\, m_l, \qquad m_l = -l,\dots,+l .$$

**Trick used here.** The real electron also has spin. To isolate the *orbital-only* effect we switch
spin off in the code: spin-orbit constant $\zeta = 0$ and spin g-factor $g_s = 0$.
"""),
("code", HEAD + r"""from src.solvers import energy_vs_field
from src.constants import MU_B_EV

b = np.linspace(0, 10, 50)                        # tesla
e = energy_vs_field(1, b, 0.0, g_s=0.0)           # l=1, zeta=0, g_s=0 -> orbital only
plt.plot(b, e * 1e3)
plt.xlabel('B (T)'); plt.ylabel('E (meV)'); plt.title('p orbital, orbital-only Zeeman effect')
plt.show()

spacing_per_tesla = (e[-1, -1] - e[-1, 0]) / 2 / b[-1]
print('level spacing per tesla (eV/T):', spacing_per_tesla)
print('Bohr magneton mu_B       (eV/T):', MU_B_EV)"""),
("markdown", r"""Three distinct lines appear (each is doubly degenerate because spin is switched off): $m_l=-1,0,+1$.
The measured spacing per tesla equals the Bohr magneton, as the formula predicts.

## The Lorentz triplet

Now let the p level decay to an s level. Selection rules (Level 4 explains why) allow
$\Delta m_l = 0, \pm 1$, so we see three lines: an unshifted $\pi$ line and two $\sigma$ lines
shifted by $\pm\mu_B B$.
"""),
("code", r"""from src.transitions import zeeman_lines
from src.animation import plot_stick_spectrum
from src.constants import MU_B_J, HBAR
import scipy.constants as sc

df = zeeman_lines(1, 0, 1.0, 0.0, 0.0, 0.0, g_s=0.0)   # B = 1 T, p -> s
print(df[['polarisation', 'wavenumber_cm1', 'intensity']].round(4).to_string(index=False))
print('mu_B B / (h c) at 1 T = %.4f cm^-1' % (MU_B_J * 1.0 / (sc.h * sc.c) / 100))

plot_stick_spectrum(df, 0.0, title='Normal Zeeman triplet at B = 1 T')
plt.show()"""),
("markdown", r"""The table lists 6 components, but they sit at only **3 distinct positions**: spin-up and spin-down give
identical shifts when spin is switched off, so they lie on top of each other. The shifts are
$-0.467,\ 0,\ +0.467\ \mathrm{cm^{-1}}$ per tesla, exactly $\mu_B B/(hc)$.

## Does it depend on $l$?

For a level with orbital momentum $l$ there are $2l+1$ equally spaced sublevels. Check $l=2$:
"""),
("code", r"""for l in (1, 2, 3):
    levels = np.unique(np.round(energy_vs_field(l, [1.0], 0.0, g_s=0.0)[0] / MU_B_EV, 6))
    print('l =', l, ' energies / (mu_B B) =', levels)"""),
("markdown", r"""**Take-away.** With spin ignored, the Hamiltonian is just $\mu_B B L_z$. Its eigenvalues are the integers
$m_l$ in units of $\mu_B B$, and the spectrum is a symmetric triplet. Real atoms usually show *more*
than three lines. That is the anomalous Zeeman effect (Level 2).
"""),
]

# --------------------------------------------------------------------------------------------
NB2 = [
("markdown", r"""# Level 2: The anomalous Zeeman effect (spin-orbit + Landé $g$)

**Goal.** Add electron spin and spin-orbit coupling, and reproduce the Landé $g$-factors that
explain why sodium's D lines split into 4 and 6 components rather than 3.

**Physics.**
$$H = \zeta\,\mathbf L\!\cdot\!\mathbf S + \mu_B B\,(g_l L_z + g_s S_z),\qquad
g_J = 1 + (g_s-1)\frac{j(j+1)+s(s+1)-l(l+1)}{2j(j+1)}$$
For weak fields ($\mu_B B \ll \zeta$): $\Delta E = g_J\,\mu_B B\,m_j$.

**Fixing $\zeta$ from experiment.** For one p electron, $E(j{=}3/2) - E(j{=}1/2) = \tfrac32\zeta$.
Sodium's measured 3p fine-structure splitting is $17.196\ \mathrm{cm^{-1}}$. We set $\zeta$ from that
single number. Everything after that is a prediction.
"""),
("code", HEAD + r"""from src.validation import zeta_from_fine_structure, fine_structure_splitting_cm1
from src.hamiltonians import landé_g
from src.solvers import energy_vs_field
from src.constants import MU_B_EV

zeta = zeta_from_fine_structure(17.196, l=1)
print('zeta = %.6f eV = %.4f meV' % (zeta, zeta * 1e3))
print('splitting recomputed from the matrix:', round(fine_structure_splitting_cm1(1, zeta), 6), 'cm^-1')
print('crossover field B* = zeta/mu_B = %.1f T' % (zeta / MU_B_EV))"""),
("markdown", r"""## Do the numerical slopes match the Landé formula?

At a small field every level moves linearly with slope $g_J m_j$ (in units of $\mu_B$).
We compute the slopes numerically and compare with the formula:
"""),
("code", r"""b = 1e-3 * zeta / MU_B_EV                                   # tiny field: mu_B B << zeta
e0 = energy_vs_field(1, [0.0], zeta)[0]
e1 = energy_vs_field(1, [b], zeta)[0]
numeric = np.sort((e1 - e0) / (MU_B_EV * b))

g12, g32 = landé_g(1, 0.5), landé_g(1, 1.5)
expected = np.sort([g12 * m for m in (-0.5, 0.5)] + [g32 * m for m in (-1.5, -0.5, 0.5, 1.5)])

print('g_J(2P1/2) = %.4f   g_J(2P3/2) = %.4f   (g_s = 2.0023)' % (g12, g32))
print('%10s %10s' % ('numeric', 'Lande'))
for n_, x in zip(numeric, expected):
    print('%10.4f %10.4f' % (n_, x))"""),
("code", r"""from src.animation import plot_energy_levels, b_scale_tesla
ax = plot_energy_levels(1, 0.3 * b_scale_tesla(zeta), zeta)
ax.set_title('Weak-field region: straight lines with slopes g_J m_j')
plt.show()"""),
("markdown", r"""## Why 10 lines instead of 3

Compare the p to s spectrum with spin switched off (Level 1) and on (this level), at the same field:
"""),
("code", r"""from src.transitions import zeeman_lines
from src.constants import wavenumber_to_ev

gap = wavenumber_to_ev(16956.17) + zeta           # puts 2P1/2 at the measured Na D1 energy
normal = zeeman_lines(1, 0, 1.0, 0.0, 0.0, gap, g_s=0.0)
anomalous = zeeman_lines(1, 0, 1.0, zeta, 0.0, gap)
distinct = lambda d: len(np.unique(np.round(d.wavenumber_cm1.to_numpy(), 4)))
print('distinct line positions, spin off (normal Zeeman)     :', distinct(normal))
print('distinct line positions, spin-orbit on (Na D1 + D2)   :', distinct(anomalous))"""),
("markdown", r"""**Take-away.** Spin-orbit coupling creates two $j$ levels with *different* $g_J$ values, so the
sublevels fan out at different rates and the lines no longer coincide. Sodium D1 gives 4 lines and D2
gives 6, which is 10 in total. The numbers above reproduce the textbook $g_J = 2/3$ and $4/3$ (small
differences come from using $g_s = 2.0023$ instead of exactly 2).
"""),
]

# --------------------------------------------------------------------------------------------
NB3 = [
("markdown", r"""# Level 3: From weak field to Paschen-Back

**Goal.** Follow the levels continuously as $B$ grows from "weak" ($\mu_B B \ll \zeta$) to "strong"
($\mu_B B \gg \zeta$), and see $\mathbf L$ and $\mathbf S$ decouple.

**Physics.** In a strong field the Zeeman term wins and $m_l, m_s$ become good quantum numbers:
$$E \simeq \mu_B B\,(m_l + g_s m_s) + \zeta\, m_l m_s .$$
The crossover field is $B^\ast = \zeta/\mu_B$.

**Numerical subtlety.** `numpy.linalg.eigh` always returns eigenvalues *sorted*. When two physical
levels cross, sorting swaps them and hides the crossing. `solvers.track_levels` follows each level
by matching eigenvectors from one field step to the next.
"""),
("code", HEAD + r"""from src.validation import zeta_from_fine_structure
from src.solvers import energy_vs_field, track_levels
from src.constants import MU_B_EV, G_S

zeta = zeta_from_fine_structure(17.196, l=1)
bstar = zeta / MU_B_EV
b = np.linspace(0, 5 * bstar, 400)
print('B* = %.1f T' % bstar)

sorted_e = energy_vs_field(1, b, zeta)
tracked = track_levels(1, b, zeta)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for k in range(6):
    axes[0].plot(b, sorted_e[:, k] * 1e3)
    axes[1].plot(b, tracked[:, k] * 1e3)
axes[0].set_title('Sorted by energy (crossings hidden)')
axes[1].set_title('Tracked levels (crossings visible)')
for a in axes:
    a.set_xlabel('B (T)'); a.axvline(bstar, color='k', ls=':', lw=1)
axes[0].set_ylabel('Energy (meV)')
plt.tight_layout(); plt.show()"""),
("markdown", r"""The dotted line marks $B^\ast$. Levels with different $m_j = m_l + m_s$ really do cross (right panel).
Levels with the *same* $m_j$ repel each other and never cross. Sorting merges both pictures into one.

## Compare with the Paschen-Back formula

Overlay the strong-field prediction on the exact curves:
"""),
("code", r"""fig, ax = plt.subplots(figsize=(7, 4.5))
for k in range(6):
    ax.plot(b, tracked[:, k] * 1e3, color='tab:blue', lw=1.5)
for ml in (1, 0, -1):
    for ms in (0.5, -0.5):
        pb = MU_B_EV * b * (ml + G_S * ms) + zeta * ml * ms
        ax.plot(b, pb * 1e3, 'r--', lw=0.8)
ax.plot([], [], color='tab:blue', label='exact diagonalisation')
ax.plot([], [], 'r--', label='Paschen-Back formula')
ax.axvline(bstar, color='k', ls=':', lw=1)
ax.set_xlabel('B (T)'); ax.set_ylabel('Energy (meV)'); ax.legend()
plt.show()"""),
("code", r"""# Slopes at a very large field (100 B*) versus the strong-field prediction (m_l + g_s m_s)
b_hi = np.array([100 * bstar, 101 * bstar])
e_hi = energy_vs_field(1, b_hi, zeta)
slopes = np.sort((e_hi[1] - e_hi[0]) / (b_hi[1] - b_hi[0]) / MU_B_EV)
pred = np.sort([ml + G_S * ms for ml in (1, 0, -1) for ms in (0.5, -0.5)])
print('%10s %10s' % ('slope/mu_B', 'ml+gs*ms'))
for s_, p_ in zip(slopes, pred):
    print('%10.4f %10.4f' % (s_, p_))"""),
("markdown", r"""The agreement improves as $B$ grows: at $5B^\ast$ the slopes are still a couple of percent off because
the spin-orbit term keeps mixing states of equal $m_j$, but at $100B^\ast$ they match the formula.

**Take-away.** Below $B^\ast$ the levels are grouped by $j$ (Level 2); far above it they are grouped by
$m_l + g_s m_s$ and the red dashed lines match. The two stretched states ($m_l = \pm1$, $m_s = \pm\tfrac12$)
are exact at every field, which the unit tests check. The animation `docs/figures/p_orbital_split.gif`
shows the same transition as a movie.
"""),
]

# --------------------------------------------------------------------------------------------
NB4 = [
("markdown", r"""# Level 4: Selection rules, line intensities and parameter fitting

**Goal.** Predict not only *where* the Zeeman lines are but also *how bright* and *how polarised*
they are, then fit a parameter back from "measured" data.

**Physics.** The electric-dipole operator is a rank-1 spherical tensor. By the Wigner-Eckart theorem
its matrix elements are proportional to a Wigner 3-j symbol:
$$\langle l'm'|C^{(1)}_q|l\,m\rangle = (-1)^{m'}\sqrt{(2l+1)(2l'+1)}
\begin{pmatrix} l' & 1 & l\\ 0&0&0\end{pmatrix}
\begin{pmatrix} l' & 1 & l\\ -m' & q & m\end{pmatrix}.$$
This enforces $\Delta l = \pm1$ and $m' - m = q \in \{-1,0,+1\}$ ($q=0$: $\pi$ light, $q=\pm1$: $\sigma$).
Intensities are the squared matrix elements between the exact eigenstates of each level.
"""),
("code", HEAD + r"""from src.transitions import dipole_matrices

d = dipole_matrices(1, 0)          # upper l=1 (6 states) to lower l=0 (2 states)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, q in zip(axes, (-1, 0, 1)):
    im = ax.imshow(d[q], cmap='RdBu', vmin=-0.7, vmax=0.7)
    ax.set_title('q = %+d' % q)
    ax.set_yticks(range(6)); ax.set_yticklabels(['(+1,up)', '(+1,dn)', '(0,up)', '(0,dn)', '(-1,up)', '(-1,dn)'], fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['s up', 's dn'], fontsize=8)
    ax.grid(False)
axes[0].set_ylabel('upper state (m_l, spin)')
plt.tight_layout(); plt.show()"""),
("markdown", r"""Each panel is the dipole operator for one polarisation. Only one orbital row is non-zero per $q$
(selection rule $\Delta m = q$) and spin is never flipped (blocks are diagonal in spin).

## Sodium D lines: positions, polarisation, intensities
"""),
("code", r"""from src.validation import zeta_from_fine_structure
from src.transitions import zeeman_lines
from src.animation import plot_stick_spectrum
from src.constants import MU_B_EV, wavenumber_to_ev, ev_to_wavenumber

zeta = zeta_from_fine_structure(17.196, l=1)
gap = wavenumber_to_ev(16956.17) + zeta           # measured Na D1 sets the energy origin
d1c, d2c = gap - zeta, gap + zeta / 2
B = 0.05                                          # weak field so the textbook patterns are clean
df = zeeman_lines(1, 0, B, zeta, 0.0, gap)
mid = 0.5 * (d1c + d2c)

for name, centre, sel in (('D1', d1c, df.energy_ev < mid), ('D2', d2c, df.energy_ev >= mid)):
    sub = df[sel].copy()
    sub['shift (mu_B B)'] = (sub.energy_ev - centre) / (MU_B_EV * B)
    print(name, '  components:', len(sub), '  total intensity: %.3f' % sub.intensity.sum())
    print(sub[['polarisation', 'shift (mu_B B)', 'intensity']].round(3).to_string(index=False))
    print()

ratio = df[df.energy_ev >= mid].intensity.sum() / df[df.energy_ev < mid].intensity.sum()
print('D2 : D1 intensity ratio = %.4f   (statistical weights 4 : 2 -> 2)' % ratio)"""),
("markdown", r"""Textbook values for sodium (LS coupling): D1 lines at $\pm\tfrac23$ ($\pi$) and $\pm\tfrac43$ ($\sigma$)
in units of $\mu_B B$ with relative intensities $1:2$; D2 lines at $\pm\tfrac13$ ($\pi$), $\pm1$ ($\sigma$),
$\pm\tfrac53$ ($\sigma$) with intensities $2:3:1$. The numbers above reproduce all of it, with no
pattern typed in by hand. They come purely from diagonalising $H$ and applying the dipole operator.
"""),
("code", r"""df1 = zeeman_lines(1, 0, 1.0, zeta, 0.0, gap)
mid = 0.5 * (d1c + d2c)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
plot_stick_spectrum(df1[df1.energy_ev < mid], ev_to_wavenumber(d1c), axes[0], 'Na D1, B = 1 T')
plot_stick_spectrum(df1[df1.energy_ev >= mid], ev_to_wavenumber(d2c), axes[1], 'Na D2, B = 1 T')
plt.tight_layout(); plt.show()"""),
("markdown", r"""## Fitting a parameter back from data

In a real experiment you measure energies and want the physical constants. Here we create
synthetic "measurements" with a known $\zeta$ and 1 micro-eV noise, then recover it with
`src.fitting.fit_parameters` (SciPy least squares), starting from a deliberately wrong guess.
"""),
("code", r"""from src.fitting import fit_parameters
from src.solvers import energy_vs_field

rng = np.random.default_rng(1)
bs = np.linspace(1, 40, 15)
data = energy_vs_field(1, bs, zeta) + rng.normal(0, 1e-6, size=(bs.size, 6))   # eV

res = fit_parameters(1, bs, data, zeta0_ev=2.0 * zeta)                         # start 2x too large
print('true zeta  = %.6f eV' % zeta)
print('fitted     = %.6f eV   (relative error %.3f %%)' % (res.zeta_ev, 100 * (res.zeta_ev - zeta) / zeta))
print('rms residual = %.2e eV' % res.rms_residual_ev)

res2 = fit_parameters(1, bs, data, zeta0_ev=zeta, g_s0=2.0, fit_g_s=True)
print('fit with g_s free:  zeta = %.6f eV,  g_s = %.5f  (true g_s = 2.00232)' % (res2.zeta_ev, res2.g_s))"""),
("markdown", r"""## Optional: the same fit with PyTorch autograd

`src/torch_engine.py` does the same job with automatic differentiation. It needs PyTorch
(`pip install torch`), which is optional for this project.
"""),
("code", r"""try:
    from src.torch_engine import fit_zeta
    z_torch = fit_zeta(1, bs, data, zeta0=1e-3)
    print('torch fit zeta = %.6f eV   (scipy fit = %.6f eV)' % (z_torch, res.zeta_ev))
except ImportError:
    print('PyTorch is not installed here, so this optional cell is skipped.')"""),
("markdown", r"""**Take-away.** Level 4 completes the chain: Hamiltonian, then energies, then transition
amplitudes, then a full predicted spectrum with polarisation and intensity. Matching the known sodium
pattern is the strongest evidence in the project that the model is right. The fitting step shows how
the same machinery works in reverse to extract constants from data.
"""),
]

# --------------------------------------------------------------------------------------------
NB5 = [
("markdown", r"""# Extension A: Hydrogen n = 2 (2s and 2p together, with the Lamb shift)

**Goal.** Put two different orbital levels, 2s and 2p, into one Hamiltonian, add the Lamb shift, and see
what a magnetic field does to the whole $n=2$ manifold. We then compare a predicted level-crossing field
with the literature.

**The eight states.** 2S$_{1/2}$ (2 states), 2P$_{1/2}$ (2 states), 2P$_{3/2}$ (4 states).

**Model.** Spin-orbit and Zeeman terms both conserve $l$, so the 8x8 Hamiltonian is two blocks:
$H = H(l{=}0) \oplus H(l{=}1)$. Two numbers come from experiment: the 2P fine-structure interval
(sets $\zeta$) and the **Lamb shift**, the small QED energy by which 2S$_{1/2}$ lies above 2P$_{1/2}$.
Ordinary quantum mechanics with the Dirac equation says they should be degenerate.
"""),
("code", HEAD + r"""import pandas as pd
from src.hydrogen import (track_n2, level_index, ev_to_mhz, mhz_to_ev, h_n2, label_states,
                          LAMB_SHIFT_MHZ, FINE_2P_MHZ, dirac_fine_structure_mhz)
from src.solvers import find_crossings
from src.constants import MU_B_EV, G_S
from src.hamiltonians import landé_g

b, e, labels, vectors = track_n2(np.linspace(1e-6, 0.5, 2500))     # tesla, eV, labels, eigenvectors

zero = labels.assign(energy_MHz=ev_to_mhz(e[0]))
print('Zero-field level scheme (energies above 2P1/2, in MHz):')
print(zero[['term', 'mj', 'energy_MHz']].round(2).to_string(index=False))"""),
("markdown", r"""The scheme is exactly what the inputs say: 2P$_{1/2}$ at 0, 2S$_{1/2}$ at the Lamb shift (1058 MHz),
2P$_{3/2}$ at the fine-structure interval (10969 MHz). The labels (term and $m_j$) were read off the
eigenvectors, not typed in.

**Cross-check of one input.** Pure Dirac theory predicts the 2P interval as $\alpha^2 \mathrm{Ry}/16$:
"""),
("code", r"""print('Dirac theory         : %.1f MHz' % dirac_fine_structure_mhz())
print('value used (measured) : %.1f MHz' % FINE_2P_MHZ)
print('difference            : %.2f %%   (QED corrections)' % (100 * (FINE_2P_MHZ / dirac_fine_structure_mhz() - 1)))"""),
("markdown", r"""## Weak field: Landé slopes for all three terms

At tiny $B$ every level should move with slope $g_J m_j$ (units of $\mu_B$):
"""),
("code", r"""bs, es, lab, _ = track_n2([1e-6, 1e-4, 2e-4])
g = {'2S1/2': landé_g(0, 0.5), '2P1/2': landé_g(1, 0.5), '2P3/2': landé_g(1, 1.5)}
rows = []
for _, r in lab.iterrows():
    slope = (es[2, r.level] - es[1, r.level]) / (MU_B_EV * 1e-4)
    rows.append((r.term, r.mj, slope, g[r.term] * r.mj))
print(pd.DataFrame(rows, columns=['term', 'mj', 'numeric slope', 'g_J * mj']).round(4).sort_values(['term', 'mj']).to_string(index=False))"""),
("markdown", r"""## The full picture

The dotted line marks the crossing of 2S$_{1/2}$($m_j=-\tfrac12$) with 2P$_{1/2}$($m_j=+\tfrac12$).
"""),
("code", r"""from src.animation import plot_n2_levels
i = level_index(labels, '2S1/2', -0.5)
j = level_index(labels, '2P1/2', +0.5)
b_cross = find_crossings(b, e[:, i], e[:, j])[0]
plot_n2_levels(b, e, labels, mark_fields_gauss=[b_cross * 1e4])
plt.show()
print('predicted crossing: %.1f gauss' % (b_cross * 1e4))
print('literature        : about 570 gauss (arXiv:1104.4421; that paper includes hyperfine structure)')"""),
("markdown", r"""**Agreement within about 1 %**, with one input (the Lamb shift) and no fitting to this crossing.

A straight-line estimate gives 566.5 G. The exact result is higher because the field also mixes
2P$_{1/2}$ with 2P$_{3/2}$, bending the 2P$_{1/2}$ level down by about 13 MHz at this field.

## The crossing field *measures* the Lamb shift

Because the crossing happens where the two levels meet, its position is set by the Lamb shift.
Scale the Lamb shift by 0.5, 1 and 2 and watch the crossing move:
"""),
("code", r"""for scale in (0.5, 1.0, 2.0):
    b2, e2, l2, _ = track_n2(np.linspace(1e-6, 0.3, 1500), lamb_mhz=LAMB_SHIFT_MHZ * scale)
    c = find_crossings(b2, e2[:, level_index(l2, '2S1/2', -0.5)], e2[:, level_index(l2, '2P1/2', 0.5)])
    print('Lamb shift x %.1f = %7.1f MHz  ->  crossing at %6.1f G' % (scale, LAMB_SHIFT_MHZ * scale, c[0] * 1e4))"""),
("markdown", r"""Doubling the Lamb shift roughly doubles the crossing field. This is why level crossings were used to
measure it.

## Every crossing between 2S and 2P levels below 0.5 T
"""),
("code", r"""import itertools
rows = []
for a, c_ in itertools.combinations(range(8), 2):
    la, lc = labels.iloc[a], labels.iloc[c_]
    if la.term == lc.term:
        continue
    for bc in find_crossings(b, e[:, a], e[:, c_]):
        rows.append((bc * 1e4, '%s (mj=%+.1f)' % (la.term, la.mj), '%s (mj=%+.1f)' % (lc.term, lc.mj)))
table = pd.DataFrame(sorted(rows), columns=['B (gauss)', 'level A', 'level B'])
print(table.round(1).head(12).to_string(index=False))
print('... %d crossings in total between different terms' % len(table))"""),
("markdown", r"""## Why crossings and not anticrossings?

In the table, levels with the *same* $m_j$ (for example 2S$_{1/2}$ and 2P$_{1/2}$ with $m_j=-\tfrac12$,
crossing near 1195 G) simply cross. In Level 3 same-$m_j$ levels repelled each other, but those were in the *same* $l$ block.
Here the two blocks are different $l$, and a magnetic field never mixes different $l$. We can prove it
from the eigenvectors:
"""),
("code", r"""worst = 0.0
for k in range(0, len(b), 20):
    s_weight = label_states(vectors[k]).s_weight.to_numpy()
    worst = max(worst, float(np.max(np.minimum(s_weight, 1 - s_weight))))
print('largest 2s/2p mixing found at any field: %.2e   (0 = never mixes)' % worst)"""),
("markdown", r"""**Take-away.** One extra block and one measured number (the Lamb shift) turn the model into a hydrogen
$n=2$ simulator. It reproduces the Landé slopes of all three terms and predicts the 2S-2P crossing field to
within about 1 % of the literature value. Real experiments (Lamb and Retherford) used an *electric* field to mix 2s
with 2p; adding that coupling is a natural next extension.
"""),
]

# --------------------------------------------------------------------------------------------
NB6 = [
("markdown", r"""# Extension B: The virtual spectrometer (colours, prism, polarisation filters)

**Goal.** Take the Zeeman lines computed in Levels 1 to 4 and put real optics in front of them:
see their colours, split light with a prism, and filter by polarisation, one component at a time.

**The chain.** source in a field $B$, then a line of sight at angle $\theta$ from $B$, then a polariser or
analyser, then a prism (colours) and/or a high-resolution stage (Zeeman components).

**An honest scale note.** A Zeeman shift is tiny: the outermost sodium D2 component moves only
about 27 pm per tesla. All the components are the *same yellow*. We will see that a prism cannot
separate them and exactly how fine an instrument has to be. Plots draw the colour of the light on a
yellow strip along the axis so the scale stays honest.
"""),
("code", HEAD + r"""import pandas as pd
from src.optics.color import wavelength_to_rgb, rainbow_strip
from src.optics.prism import refractive_index, deviation_deg, min_deviation_deg, screen_position_m
from src.optics.spectrometer import (sodium_d_components, observe, required_resolving_power,
                                     LAMP_LINES, vacuum_cm1_to_air_nm)

plt.figure(figsize=(8, 1))
plt.imshow(rainbow_strip(380, 750, 600, 30), aspect='auto', extent=[380, 750, 0, 1])
plt.yticks([]); plt.xlabel('wavelength (nm)'); plt.title('Visible spectrum as the code draws it'); plt.show()

print('Colour of sodium D light (589 nm), RGB =', tuple(round(c, 2) for c in wavelength_to_rgb(589.3)))
print(LAMP_LINES.assign(rgb=[tuple(round(c, 2) for c in wavelength_to_rgb(w)) for w in LAMP_LINES.wavelength_nm]).to_string(index=False))"""),
("markdown", r"""## Glass bends colours differently

A prism works because the refractive index depends on wavelength. The code uses the Sellmeier
equation with catalogue constants for two glasses:
"""),
("code", r"""wl = np.array([404.7, 486.1, 546.1, 589.3, 656.3])
tab = pd.DataFrame({'wavelength (nm)': wl,
                    'n (BK7)': refractive_index(wl, 'BK7'),
                    'n (SF11)': refractive_index(wl, 'SF11')}).round(5)
print(tab.to_string(index=False))
print()
print('BK7 index at the d line (587.6 nm): %.5f  (catalogue 1.5168)' % refractive_index(587.56, 'BK7'))
print('Minimum deviation of a 60 deg SF11 prism at 589 nm: %.2f deg' % min_deviation_deg(589.3, 60, 'SF11'))"""),
("code", r"""from src.optics.figures import fig_prism_setup
fig_prism_setup(); plt.show()"""),
("markdown", r"""Violet bends most, red least. The lower panel shows what a screen 2 m away would show for a lamp with
sodium, mercury and hydrogen lines: well separated colours, but the sodium D1 and D2 lines sit almost on
top of each other because they are only 0.6 nm apart.

## Why a prism cannot show the Zeeman components

Compare how far apart things land on the screen, in micrometres:
"""),
("code", r"""c = sodium_d_components(1.0)                                   # B = 1 tesla
x_um = screen_position_m(c.wavelength_nm.to_numpy(), 2.0, glass='SF11') * 1e6
d2 = c.line == 'D2'
print('Na D2 components at 1 T, spread on a screen 2 m away (SF11 prism): %.1f micrometres' % (np.ptp(x_um[d2])))
sep = screen_position_m(float(vacuum_cm1_to_air_nm(16956.17)), 2.0, glass='SF11') - screen_position_m(float(vacuum_cm1_to_air_nm(16973.37)), 2.0, glass='SF11')
print('Na D1 vs D2 (0.6 nm apart):                                       %.1f micrometres' % (abs(sep) * 1e6))
print()
airy_um = 1.22 * 589.3e-9 * 2.0 / 0.01 * 1e6          # diffraction-limited spot of a 1 cm beam focused at 2 m
print('Smallest focused spot of a 1 cm wide beam at 2 m (diffraction limit): about %.0f micrometres' % airy_um)
print()
print('The whole D2 Zeeman pattern is spread over less than that spot, so the components blur together.')
print('A prism of this kind cannot show them; a finer instrument is needed.')"""),
("markdown", r"""## The instrument ladder

Resolving power $R=\lambda/\Delta\lambda$ says how fine an instrument can see. The closest sodium Zeeman
components need a certain $R$ at each field; instruments provide a certain $R$. Where they meet is the
weakest field at which the pattern is visible:
"""),
("code", r"""from src.optics.instruments import fig_instrument_ladder, instrument_ladder
from src.optics.spectrometer import field_for_resolving_power

print('Resolving power needed to separate the closest Zeeman components of Na:')
for b in (0.1, 0.5, 1.0, 2.0, 5.0):
    print('  B = %4.1f T : R = %8.0f' % (b, required_resolving_power(sodium_d_components(b))))
print()
rows = [(k, round(v), round(field_for_resolving_power(v), 3)) for k, v in instrument_ladder().items()]
print(pd.DataFrame(rows, columns=['instrument', 'ideal R', 'weakest resolvable field (T)']).to_string(index=False))
fig_instrument_ladder(); plt.show()"""),
("markdown", r"""**Take-away.** A 5 cm glass prism (R about 2000) would only reveal the pattern at about 25 T, where the weak-field
picture has already broken down. A large grating needs roughly 0.5 T, a Fabry-Perot about 0.05 T. This is why Zeeman
spectroscopy uses high-resolution instruments, and why the spectrum below is drawn in picometres, not as colours.

## Polarisation: what each viewing direction shows

A field-aligned atom radiates three kinds of light. With $q=m_{upper}-m_{lower}$:
$\pi$ ($q=0$) is a dipole along $B$, intensity $\propto\sin^2\theta$; $\sigma^\pm$ ($q=\pm1$) is a dipole rotating around $B$,
intensity $\propto(1+\cos^2\theta)/2$, where $\theta$ is the angle between line of sight and $B$.
Across the field the $\pi$ light is linear along $B$ and the $\sigma$ light linear across it; along the field only
circular $\sigma$ light remains.
"""),
("code", r"""from src.optics.figures import fig_polarisation_ellipses, fig_polarisation_filters
fig_polarisation_ellipses(); plt.show()"""),
("code", r"""fig_polarisation_filters(2.0); plt.show()"""),
("markdown", r"""Top row: looking across $B$. A polariser parallel to $B$ keeps only $\pi$ lines; one perpendicular keeps only $\sigma$.
Bottom row: looking along $B$ there are no $\pi$ lines, and a circular analyser (quarter-wave plate plus polariser)
separates $\sigma^+$ (red) from $\sigma^-$ (green).

## Check against the textbook: classic intensity ratios

Viewed across the field in weak field, the four D1 components are equally bright, and the D2 components
have relative intensities 4 : 3 : 1 (the $\pi$ lines : inner $\sigma$ : outer $\sigma$). Nothing here is typed in:
it comes from the matrix elements and the angular pattern.
"""),
("code", r"""comps = sodium_d_components(0.05)
seen = observe(comps, 90.0)
d1 = seen[seen.line == 'D1'].observed.to_numpy()
print('D1 observed intensities:', np.round(d1, 3), '  (all equal)')
d2 = seen[seen.line == 'D2'].assign(absshift=lambda d: d.shift_pm.abs()).sort_values('absshift')
pi, s_in, s_out = d2.observed.to_numpy()[[0, 2, 4]]
print('D2: pi : inner sigma : outer sigma = %.2f : %.2f : %.2f' % (pi / s_out, s_in / s_out, 1.0))"""),
("markdown", r"""## Animations

Generated by `scripts/make_optics_figures.py --gif` into `docs/figures/`.

**1. Colour filter scan.** A colour (band-pass) filter sweeps across the lamp spectrum and only the lines inside its
pass band reach the screen. This is "filter each colour one by one".

![filter scan](../docs/figures/optics_filter_scan.gif)

**2. Field ramp.** Raising $B$ makes the Zeeman components of Na D1 and D2 fan out.

![field ramp](../docs/figures/optics_zeeman_field_ramp.gif)

**3. Rotating polariser.** Across the field, $\pi$ and $\sigma$ lines trade places as the polariser turns (Malus's law).

![polariser rotation](../docs/figures/optics_polariser_rotation.gif)

**4. Tilting the line of sight.** From across the field to along it: $\pi$ fades out and $\sigma$ becomes circular.

![view angle](../docs/figures/optics_view_angle.gif)
"""),
("markdown", r"""**Take-away.** Colour tells you the *element* and the *transition*; the Zeeman components live far finer than
the eye or a prism can see; and polarisation is the tool that sorts them: linear filters across the field, circular
filters along it. Together they let you pick out single components and compare their intensities with theory.
"""),
]

NB7 = [
("markdown", r"""# Extension C: Hydrogen H-alpha in a magnetic field

**Goal.** Give hydrogen a real *spectrum*. The n = 2 lab (notebook 5) only has levels. Here atoms jump from
n = 3 (18 states with spin) to n = 2 (8 states), which makes the red Balmer line H-alpha at 656 nm. The result goes
through the same polarisation filters as sodium.

**What is new compared with sodium.** Several l values live in one shell, so the Hamiltonian is block-diagonal in l.
Fine structure comes from Dirac theory, the Lamb shift lifts the s levels, and the strength of every jump needs a
*radial integral* computed from the hydrogen wavefunctions.

**Not modelled:** hyperfine structure, the diamagnetic term (about 0.02 cm^-1 at 20 T) and the exact n = 3 Lamb shift
(scaled from n = 2 as 1/n^3).
"""),
("code", HEAD + r"""from src.hydrogen_spectrum import (radial_integral, einstein_a, balmer_components, doppler_fwhm_pm)
from src.optics.spectrometer import observe
from src.optics.polarization import linear_polariser
print('<2p|r|1s> = %.6f a0   exact 128*sqrt(6)/243 = %.6f' % (radial_integral(2,1,1,0), 128*6**.5/243))
for (nu,lu,nl,ll), ref in {(2,1,1,0):6.265e8,(3,0,2,1):6.317e6,(3,1,2,0):2.245e7,(3,2,2,1):6.465e7}.items():
    print('A(%d%s -> %d%s) = %.4g /s   (reference ~%.4g)' % (nu,'spdf'[lu],nl,'spdf'[ll],einstein_a(nu,lu,nl,ll),ref))"""),
("markdown", r"""## Check 1: the radial integral and decay rates

The Lyman-alpha integral has an exact value and ours agrees to 8 digits. The decay rates come from the same integrals and
agree with the literature values quoted from memory (check against NIST before citing).
"""),
("code", r"""df = balmer_components(0.001)
print('components at 1 mT:', len(df), ' distinct wavelengths:', df.wavelength_nm.round(5).nunique())
print('zero-field H-alpha is a cluster of fine-structure lines within about %.0f pm:' % (df.shift_pm.max()-df.shift_pm.min()))
print('thermal Doppler width at 300 K: %.1f pm' % doppler_fwhm_pm())"""),
("markdown", r"""## Check 2: the strong-field limit

At high field the pattern must become the *normal* Zeeman triplet, with sigma components at plus or minus 20.11 pm per tesla.
"""),
("code", r"""for b in (2, 6, 20):
    d = balmer_components(b, threshold=1e-9)
    m = lambda q: np.average(d[d.q==q].shift_pm, weights=d[d.q==q].rate)
    print('B = %2d T: sigma half-separation = %.2f pm   normal Zeeman = %.2f pm' % (b, (m(-1)-m(1))/2, 20.11*b))"""),
("markdown", r"""## Seeing it: three regimes and three filters

Rows: no filter, polariser parallel to B (keeps pi), polariser perpendicular to B (keeps sigma). The grey curve is what a
room-temperature lamp would really show once Doppler broadening blurs the sticks.

![regimes](../docs/figures/halpha_regimes.png)

At 0.05 T the cluster of fine-structure lines just starts to split. At 1 T the pattern is a mess, because the Zeeman and fine-structure
energies are comparable. At 6 T the shell is in the Paschen-Back regime: two sigma groups at about plus and minus 120 pm and a pi group in the middle.
"""),
("code", r"""fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
comps = balmer_components(6.0)
for ax, (name, m) in zip(axes, [('no filter', None), ('parallel to B', linear_polariser(0)), ('perpendicular to B', linear_polariser(90))]):
    d = observe(comps, 90.0, m)
    d = d[d.observed > 0.003*d.observed.max()]
    ax.vlines(d.shift_pm, 0, d.observed, color=d.polarisation.map({'pi':'tab:blue','sigma+':'tab:red','sigma-':'tab:green'}))
    ax.set_title(name); ax.set_xlabel('shift (pm)')
axes[0].set_ylabel('intensity')
plt.tight_layout(); plt.show()"""),
("markdown", r"""## The fan diagram

![fan](../docs/figures/halpha_fan.png)

The sigma components approach the dashed normal-Zeeman lines and the pi components stay near the centre.

**Take-away.** The same machinery (diagonalise, apply the dipole operator, push through Jones matrices) works for a second atom with
a richer shell structure. Only the Hamiltonian and the radial factor changed.
"""),
]

NOTEBOOKS = {
    "01_normal_zeeman_lab.ipynb": NB1,
    "02_anomalous_zeeman_lab.ipynb": NB2,
    "03_paschen_back_transition_lab.ipynb": NB3,
    "04_tensor_operators_torch.ipynb": NB4,
    "05_hydrogen_n2_lab.ipynb": NB5,
    "06_virtual_spectrometer_lab.ipynb": NB6,
    "07_hydrogen_halpha_lab.ipynb": NB7,
}

for name, cells in NOTEBOOKS.items():
    (OUT / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", name)
