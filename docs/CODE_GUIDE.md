# Code guide: what every script does and why

This guide explains each file in the project, in the order the physics is built. Read it next to
the code. Energies are in **eV**, fields in **tesla**, and angular momentum operators use ħ = 1.

## The big picture

```
constants.py ──► hamiltonians.py ──► solvers.py ──► animation.py / validation.py
                                 └──► transitions.py (Level 4) ──► fitting.py
```

1. **Build** the Hamiltonian as a matrix (`hamiltonians.py`).
2. **Diagonalise** it for many field values (`solvers.py`). The eigenvalues are the energy levels.
3. **Predict spectra** from the eigenvectors (`transitions.py`).
4. **Plot** (`animation.py`) and **compare with experiment** (`validation.py`, `data/`).
5. **Fit** parameters back from data (`fitting.py`).

---

## `src/constants.py`: physical constants

Takes values from `scipy.constants` (CODATA), so no number is typed by hand.

| Name | Meaning |
|---|---|
| `MU_B_EV` | Bohr magneton in eV per tesla (5.788e-5). Sets the size of every Zeeman shift. |
| `G_S` | Electron spin g-factor, 2.0023. |
| `G_L` | Orbital g-factor, 1. |
| `HC_EV_CM` | h·c in eV·cm, to convert wavenumbers to energy. |
| `wavenumber_to_ev(σ)` | Converts cm⁻¹ to eV (spectroscopists quote lines in cm⁻¹). |
| `ev_to_wavenumber(E)` | The reverse. |

## `src/hamiltonians.py`: the Hamiltonian as a matrix

The basis is |m_l⟩ ⊗ |m_s⟩ with m_l descending (outer) and m_s = +½, −½ (inner). For l = 1 that is
6 states. Every other file relies on this ordering.

- **`angular_momentum_ops(j)`** builds J_x, J_y, J_z as matrices for any j. J_z is diagonal (the m values);
  the raising operator J₊ has entries √(j(j+1) − m(m+1)); then J_x = (J₊+J₋)/2 and J_y = (J₊−J₋)/2i.
  The tests check the commutation relation [J_x, J_y] = iJ_z, so these matrices are known to be right.
- **`coupled_ops(l)`** makes L and S act on the combined space using Kronecker products (`np.kron`):
  L acts on the orbital part and does nothing to spin, and S the opposite.
- **`h_spin_orbit(l, ζ)`** returns ζ·(L·S) = ζ(L_xS_x + L_yS_y + L_zS_z).
- **`h_zeeman(l, B)`** returns μ_B·B·(g_l L_z + g_s S_z).
- **`h_total(l, B, ζ, e0)`** adds H₀ (a constant energy), spin-orbit and Zeeman.
- **`landé_g(l, j)`** is the analytic Landé formula, used as an independent check on the numerics.

## `src/solvers.py`: from matrices to energy levels

- **`solve(l, B, ζ)`** diagonalises one Hamiltonian with `numpy.linalg.eigh` and returns eigenvalues and eigenvectors.
  `eigh` is used because H is Hermitian, which makes it fast and keeps the energies exactly real.
- **`energy_vs_field(l, B_values, ζ)`** loops over fields and returns only the eigenvalues, always sorted.
- **`track_levels(...)`** solves the *crossing problem*. Sorted eigenvalues swap identity when two
  levels cross, so plots look wrong. Here, after each field step, the new eigenvector with the biggest
  overlap with each old eigenvector is the "same" level. That gives smooth physical curves.
- **`zeeman_transitions(...)`** is a small helper returning all energy differences between two level sets.

## `src/transitions.py`: Level 4, spectra

- **`wigner_3j(...)`** implements the Racah formula for the 3-j symbol, written from scratch so no
  extra library is needed. It is tested against known values such as (1 1 0; 0 0 0) = −1/√3.
- **`dipole_matrices(l_up, l_low)`** builds the dipole operator for q = −1, 0, +1 using the
  Wigner-Eckart formula. Spin is conserved, so blocks are diagonal in spin. Zeros in this matrix *are* the selection rules.
- **`zeeman_lines(...)`** is the main function. It diagonalises the upper and lower manifolds, projects the
  dipole operator onto the exact eigenvectors (`vu† D vl`), squares the result to get intensity,
  and returns a table (energy, wavenumber, polarisation, intensity) with one row per allowed line.
  `gap_ev` places the upper manifold at its real energy above the lower one.
- **`shift_in_bohr_units(...)`** converts line positions to multiples of μ_B·B, to compare with textbook patterns.

## `src/validation.py`: comparing with experiment

- **`load_reference()`** reads `data/experimental_reference.csv`.
- **`zeta_from_fine_structure(Δ, l)`** turns a measured splitting into ζ using E(j=l+½) − E(j=l−½) = ζ(l+½).
  This is the **only parameter taken from experiment**; everything else is predicted.
- **`fine_structure_splitting_cm1(l, ζ)`** goes the other way, from the matrix, to prove the round trip.
- **`compare(...)`** returns absolute and relative errors between computed and reference values.

## `src/fitting.py`: parameters from data

- **`fit_parameters(l, B_values, measured, ζ₀)`** uses `scipy.optimize.least_squares` to adjust ζ (and optionally g_s)
  until the computed levels match the measured ones. Parameters are rescaled to O(1) so the optimiser behaves.
  The tests recover ζ to better than 1 % from noisy synthetic data.

## `src/animation.py`: plots and animation

- **`plot_energy_levels(...)`** static energy-versus-field diagram built on `track_levels`.
- **`animate_splitting(...)`** the same curves drawn progressively with `FuncAnimation`. Saved as a GIF with `save_path`.
- **`plot_stick_spectrum(...)`** a stick spectrum (position = shift, height = intensity, colour = polarisation).
- **`b_scale_tesla(ζ)`** returns B* = ζ/μ_B, the field where weak-field behaviour turns into Paschen-Back.

## `src/torch_engine.py` (optional)

The same ζ fit as `fitting.py`, but with PyTorch automatic differentiation. It only runs if PyTorch is installed.
It is meant as a starting point for larger models (many parameters), where gradients are valuable.

---

## `tests/`: why you can trust the numbers

| File | What it proves |
|---|---|
| `test_matrix_properties.py` | Operators obey angular-momentum algebra; H is Hermitian; eigenvectors are orthonormal; J_z is conserved. |
| `test_physical_constants.py` | Bohr magneton value; Landé g-factors; weak-field slopes = g_J·m_j; Paschen-Back limit; exact stretched states. |
| `test_transitions.py` | 3-j values; selection rules; parity forbids s→s; Na D1/D2 have 4/6 components at the textbook positions and intensities; D2:D1 = 2. |
| `test_fitting.py` | Known ζ and g_s are recovered from synthetic data. |

Run them all with `python -m pytest -q`.

## `scripts/`

- **`make_figures.py`** regenerates every figure in `docs/figures/` (add `--gif` for the animation).
- **`make_notebooks.py`** writes the four notebooks from their source text.
- **`execute_notebooks.py`** runs all notebook cells and saves the printed output and plots inside the `.ipynb` files.

## `notebooks/`

One lab per level, each with theory in markdown, code, and saved output.
1. Normal Zeeman and the Lorentz triplet · 2. Spin-orbit, Landé g and ζ from sodium ·
3. Weak field to Paschen-Back, with tracked levels · 4. Selection rules, line intensities and fitting.

## `docs/` and `.github/`

- `docs/main.tex` includes one `.tex` file per level; build with `cd docs && latexmk -pdf main.tex`.
- `.github/workflows/latex.yml` runs the tests, executes the notebooks and builds the PDF on every push.

---

## How to add a new system (the method)

1. Write the Hamiltonian pieces in `hamiltonians.py` (or a new file).
2. Diagonalise with `solvers.py` and plot with `animation.py`.
3. Find a measured number, add it to `data/experimental_reference.csv`, and write a test that compares it to your output.
4. Only when the test passes, move on to the next level.
