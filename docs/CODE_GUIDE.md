# Code guide: what every script does and why

This guide explains each file in the project, in the order the physics is built. Read it next to
the code. Energies are in **eV**, fields in **tesla**, and angular momentum operators use ħ = 1.

## The big picture

Where to find things (the rest of this guide follows the same order):

| You want | Look in |
|---|---|
| The core Hamiltonian, levels, spectra, fitting | `constants`, `hamiltonians`, `solvers`, `transitions`, `fitting`, `validation` |
| Hydrogen (Lamb shift, H-alpha, electric field) | `hydrogen`, `hydrogen_spectrum`, `stark` |
| Calcium (two electrons) | `two_electron`, `calcium_lines` |
| Helium 1s2s / 1s2p | `helium` |
| Gradients (PyTorch) | `torch_voigt`, `torch_engine` |
| Prism, polarisers, Fabry-Perot, level diagram | `optics/` |
| Nuclear spin and line shapes | `hyperfine`, `hyperfine_lines`, `line_profiles` |
| Small corrections, photon operators | `diamagnetic`, `isotope`, `cavity` |
| How far to trust a number | `uncertainty`, `uncertainty_bands`, `budget` |
| Comparison with a published calculation | `literature` |


```
constants.py ──► hamiltonians.py ──► solvers.py ──► animation.py / validation.py
                                 ├──► transitions.py (Level 4) ──► fitting.py
                                 ├──► hydrogen.py (Extension A: n = 2, 2s + 2p)
                                 └──► optics/ (Extension B: colours, prism, polarisation filters, instruments)
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
- **`track_from_builder(build, B_values)`** solves the *crossing problem* for **any** Hamiltonian
  (`build(B)` returns the matrix). Sorted eigenvalues swap identity when two levels cross, so plots look
  wrong. Here, after each field step, the new eigenvector with the biggest overlap with each old eigenvector
  is the "same" level. That gives smooth physical curves. It can also return the eigenvectors.
- **`track_levels(l, B_values, ζ)`** is the single-electron version: it calls `track_from_builder` with `h_total`.
- **`find_crossings(B, y_i, y_j, include_touching=False)`** finds the field values where two tracked curves cross (sign
  change of the difference, with linear interpolation). It also catches a crossing that lands exactly on a grid point
  (difference exactly 0, sign change across it) a run of two or more exact zeros is a degeneracy region, not a crossing, and is excluded and returned instead by `find_zero_intervals` as `(B_start, B_end)`; a tangential touch (zero without
  a sign change) is not a crossing and is returned only with `include_touching=True`. Two crossings closer than the grid
  spacing cannot be resolved by any sign test, so the grid must be finer than their separation.
- **`refine_crossings_from_builder(build, B, i, j, xtol=1e-12)`** is the version for final numbers: each bracket found on the
  grid is zoomed by re-diagonalising the Hamiltonian inside it (9 points per step, level identity kept by overlap), until
  the bracket is narrower than `xtol`. Nothing is interpolated, so the result does not depend on the grid. Crossings that sit exactly on a grid point are returned as they are. The practical accuracy floor is about 1e-13 T here: the eigenvalue rounding (~1e-17 eV) divided by the gap slope (7.5e-5 eV/T), so `xtol=1e-12` sits just above the noise. A sign test only finds crossings that change the sign of E_i - E_j between samples; it does not prove that every crossing in the sweep was found.
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

## `src/hydrogen.py`: Extension A, hydrogen n = 2

Puts **two different orbital levels** (2s and 2p) in one Hamiltonian. Energies are measured from 2P₁/₂ at zero field.

- **Constants:** `LAMB_SHIFT_MHZ` (1057.845) and `FINE_2P_MHZ` (10969.13) are the two numbers taken from experiment (sources in the PDF's Sources section).
  `mhz_to_ev` / `ev_to_mhz` convert between a frequency (what atomic physicists quote) and an energy.
- **`dirac_fine_structure_mhz()`** gives the pure-theory value α²·Ry/16 (about 10949 MHz). It differs from the measured
  interval by 0.2 % (QED), so it is an independent sanity check on `FINE_2P_MHZ`.
- **`h_n2(B)`** builds the 8×8 Hamiltonian as two blocks glued with `scipy.linalg.block_diag`: the l = 0 block (2S₁/₂, raised by the
  Lamb shift) and the l = 1 block (spin-orbit + Zeeman, ζ set from the 2P interval). Both terms conserve l, so the blocks never mix.
- **`label_states(vectors)`** names each eigenvector: which block (2s or 2p), m_j from ⟨J_z⟩ (exact), and j from ⟨J²⟩.
  The labels are *measured from the eigenvectors*, not typed in.
- **`track_n2(B_values, ...)`** returns tracked energies, labels and eigenvectors for a sweep. The first field must be small but non-zero,
  because at exactly B = 0 degenerate levels have arbitrary eigenvectors.
- **`level_index(labels, term, mj)`** looks up a level's column, e.g. `level_index(labels, "2S1/2", -0.5)`.

## `src/optics/fabry_perot.py`: the instrument that resolves the pattern

- `reflectivity_for_finesse(F)`: mirror reflectivity that gives finesse F (solved exactly).
- `airy(theta, wavelength, gap, R)`: transmission of the etalon for one wavelength.
- `ring_angle(n, gap)`: angle of the n-th ring.
- `radial_profile(theta, components, gap, R)`: sum of the Airy patterns of all components, weighted by observed intensity.
- `fig_fabry_perot_rings()` (ring images at three fields, with and without a polariser) and `fig_fabry_perot_profile()` (six resolved peaks).

## `src/optics/grotrian.py`: level diagram with transition arrows

- `grotrian_data(B)`: upper and lower levels (energies from each manifold's centre, quantum number M = ⟨J_z⟩) and the table of drawable arrows from `transitions.zeeman_lines`; transitions with a vanishing matrix element are dropped.
- `draw_grotrian(ax_up, ax_low, data)`: draws one frame; arrow colour = polarisation, arrow width = intensity.
- `fig_grotrian(b_values)` (static, three fields) and `animate_grotrian(path)` (GIF as B rises to 60 T).

## `src/optics/sodium_paschen_back.py`: the sodium doublet at very high field

- `components_vs_cog(B)`: the sodium D components with shifts measured from the doublet's centre of gravity (needed because D1 and D2 cross over above about 25 T).
- `fig_sodium_paschen_back_fan()`: positions against field up to 60 T, with the crossover field B* marked.
- `animate_sodium_paschen_back()`: GIF of the collapse through three polariser views.

## `src/optics/halpha_instruments.py`: can an instrument separate the H-alpha groups?

- `halpha_instrument_ladder()`: ideal resolving powers of prisms, gratings and a Fabry-Perot at 656 nm.
- `required_power_halpha(B)` and `weakest_field_halpha(R)`: R needed to separate the sigma groups (+-20.11 pm/T x B) from the pi group, and its inverse.
- `doppler_limited_power(T)`: the resolving power (about 81 000 at 300 K) beyond which thermal motion, not the instrument, sets the sharpness.
- `prism_zeeman_deflection_urad(B, glass)`: how far apart (in microradians) a prism sends the two sigma groups.
- `fig_halpha_instrument_ladder()`: the figure, with the Doppler floor drawn.

## `src/optics/`: Extension B, the virtual spectrometer

A package with one file per idea. It takes the Zeeman components from `transitions.zeeman_lines` and puts
optics in front of them: colours, a glass prism, polarisation filters, and the question of how fine an
instrument must be. Chain: **source in field B → line of sight at angle θ from B → polariser/analyser → prism and/or
high-resolution stage.**

**`color.py`**
- `wavelength_to_rgb(nm)` gives the display colour of a visible wavelength (a piecewise fit; black outside 380 to 750 nm).
  It is an approximation for plots, not a colour standard.
- `rainbow_strip(...)` builds an RGB image of the visible spectrum, used as a plot background.

**`prism.py`**
- `refractive_index(nm, glass)` is the Sellmeier equation with catalogue constants for `"BK7"` and `"SF11"`.
- `deviation_deg(nm, incidence, apex, glass)` applies Snell's law at both faces and returns the total bend of a ray.
- `min_deviation_incidence_deg` and `min_deviation_deg` give the symmetric (minimum-deviation) setting,
  δ_min = 2 asin(n sin(A/2)) − A.
- `screen_position_m(nm, distance)` converts the bend into a position on a screen, relative to the 589 nm line.
- `prism_resolving_power(nm, base_mm, glass)` is R = b·|dn/dλ|, the finest wavelength difference the prism can separate.

**`polarization.py`** (Jones calculus; the conventions are written at the top of the file)
- `component_jones(q, theta)` is the Jones vector of a component with q = m_upper − m_lower seen at angle θ from B.
  Its squared length is the brightness pattern: sin²θ for π, (1 + cos²θ)/2 for σ.
- `angular_intensity(q, theta)` returns that squared length.
- `linear_polariser(phi)`, `quarter_wave_plate(angle)` and `circular_polariser(±1)` are the 2×2 matrices of the optical elements.
- `transmitted_intensity(jones, element)` is |M·J|² after an element (or product of elements).
- `polarisation_ellipse(jones)` gives the path traced by the electric-field tip, for drawing.

**`spectrometer.py`**
- `LAMP_LINES` is a small lamp table (Hg, H and Na lines) used for colour demonstrations. The relative intensities are illustrative.
- `vacuum_cm1_to_air_nm` converts wavenumbers to air wavelengths.
- `sodium_d_components(B)` runs the atomic model and returns every D1/D2 component with its polarisation,
  air wavelength, shift in pm from the zero-field line, and emission rate. B must be above zero.
- `observe(components, theta, element)` multiplies each rate by |M·J|²: this is the intensity an observer sees at that
  angle through that filter.
- `bandpass_transmission` is a colour filter; `gaussian_spectrum` blurs lines with an instrument profile.
- `required_resolving_power(components)` is the λ/Δλ needed to separate the two closest components;
  `field_for_resolving_power(R)` inverts it: the weakest field at which an instrument of power R just separates them.

**`instruments.py`**
- `grating_resolving_power` (R = m·N) and `fabry_perot_resolving_power` (R ≈ m·F) add two instrument types to the prism.
- `instrument_ladder()` lists ideal resolving powers from a prism up to a Fabry-Perot. Real instruments reach less.
- `required_power_vs_field(B_values)` is the power the sodium pattern needs at each field.
- `fig_instrument_ladder()` draws both comparisons. The result: a 5 cm BK7 prism would need about 26 T, a large grating 0.45 T.

**`figures.py`**: static figures `fig_prism_setup`, `fig_polarisation_filters`, `fig_polarisation_ellipses`, and four animations
`animate_filter_scan` (a colour filter sweeping across the lamp spectrum), `animate_zeeman_field_ramp`,
`animate_polariser_rotation` (Malus's law) and `animate_view_angle` (π fades out, σ becomes circular).

## `src/two_electron.py`: two valence electrons (calcium 4s4p)

- `operators()` and `h_two_electron(B, zeta, delta, E_t)`: the 12x12 Hamiltonian: exchange splitting, spin-orbit coupling of the p electron, Zeeman term.
- `classify_states(B, ...)`: eigenstates labelled 3P0, 3P1, 3P2 or 1P1 by their J and singlet weight.
- `zero_field_levels_cm1(...)` and `fit_to_levels(...)`: term energies from the model, and the fit of the three parameters to four measured energies.
- `singlet_admixture_in_3p1(...)`: how much 1P1 is mixed into 3P1 (the reason the 657 nm intercombination line is weakly allowed).
- `g_factor(label, ...)`: g_J from the model's own eigenvalues at a tiny field.
- `intercombination_rate(...)`: rough decay rate of 3P1 from the admixture. `CA_LEVELS_CM1`: calcium energies from the NIST Handbook (³P₁ from a measured frequency).
- `two_electron_figures.py`: `fig_two_electron_levels(zeta, delta, e_t)` (levels and Zeeman fans against field).

## `src/literature.py`: cross-check against a published calculation

- `PAPER_AT_0P9T`: the reference table typed in from arXiv:2104.14896 (five group spacings, formula and numerical values, with how each was obtained).
- `model_group_spacings_ghz(B)`: the same spacings measured from our model's lines.
- `comparison_table()`: model against paper, with percentage differences; `fig_literature_comparison()` plots them.
- `MEASURED_G_J`, `model_g_j_from_diagonalisation()` and `g_factor_comparison()`: measured sodium Landé factors (Steck's table) against g_J read from the model's own eigenvalues at a tiny field.
- Because the paper includes hyperfine structure and ours does not, only the spacings between groups are compared.

## `src/hyperfine.py`: the nucleus joins in (sodium, hydrogen)

- Basis |l m_l⟩⊗|s m_s⟩⊗|I m_I⟩, so hydrogen n = 2 is 16 states, sodium 3S 8 states and 3P 24 states.
- `electron_projectors(l)`: projectors onto each J from the eigen-decomposition of J², so each fine-structure level gets its own constant A_J.
- `h_hyperfine(l, i, a_by_j_ev, B, g_i)`: Σ A_J P_J (I·J) P_J plus the nuclear Zeeman term μ_B g_I I_z B.
- `h_with_hyperfine`, `h_n2_hyperfine(B)`, `h_na_3s(B)`, `h_na_3p(B, zeta)`: ready-made Hamiltonians.
- `lamb_crossings_gauss(hyperfine=True/False)`: the 2S × 2P crossing fields near 573 G (one without hyperfine, four with).
- Constants: sodium from Steck's table; hydrogen 2S measured; hydrogen 2P leading-order theory (marked in the code).
- `src/hyperfine_lines.py`: `sodium_d_hyperfine_lines(B)` (every D1/D2 component: line, q, polarisation, frequency in MHz from the hyperfine-free D1 line, strength; the dipole acts as D ⊗ 1_I), `group_lines` (merge degenerate components at B = 0), `doppler_fwhm_mhz(T)`, `broadened_spectrum`, `fig_na_d2_hyperfine()`. Hydrogen H-alpha is left out on purpose: its n = 3 hyperfine constants are not available from a checked source, and the splittings (MHz) are far below the 7 GHz Doppler width.
- `src/line_profiles.py`: `natural_fwhm_mhz(A_up, A_low)`, `voigt_mhz(freq, centre, fwhm_gauss, fwhm_lorentz)` (area 1), `voigt_fwhm_approx` and `voigt_fwhm_exact`, `voigt_spectrum(lines, line, fwhm_gauss, fwhm_lorentz, freq)` (one Voigt per component, area = strength), `fig_voigt()`. Pressure width is an input you choose; no collision coefficient is built in.
- `src/diamagnetic.py`: `r2_hydrogenic(n, l)`, `sin2_theta_mean(l, m)`, `diamagnetic_shift_ev(n, l, m, B)` (first-order estimate; not added to the Hamiltonians), `balmer_alpha_diamagnetic_shift_cm1(B)`.
- `src/isotope.py`: `rydberg_cm1('H' or 'D')`, `line_wavenumber_cm1`, `isotope_shift()` (D minus H, mass effect only), `zeeman_shift_pm_per_t`.
- `src/corrections_figures.py`: `fig_corrections()`.
- `src/calcium_lines.py`: `dipole_state(q)` (singlet with m_l = q), `calcium_lines(B, zeta, delta, e_t)` (DataFrame: label, wavenumber_cm1, wavelength_nm, q, strength; strengths are |<k|d_q 1S0>|², summing to 1 for each q), `term_strength(B, label, ...)`, `fig_calcium_lines(B)`.
- `src/helium.py`: `h_exchange(G)`, `h_spin_spin(rho)`, `h_helium_1s2p(B, G, zeta_S, rho, zeta_A, e_dir)` (12x12), `h_helium_1s2s(B, G)` (4x4), `triplet_p_levels`, `fit_triplet_p_intervals(e01, e12)`, `classify` (columns: energy_ev, mj, m_j, singlet, j2, p_j0..p_j2, j = dominant J, j_purity; J is only a good label at B = 0), `classify_tracked(B, build)` (follows every level from B -> 0 and labels it by its zero-field parent, e.g. 2^3P_1; use this for finite-field figures), `slater_integral`, `exchange_g_hartree`, `direct_f_hartree`.
- `src/helium_data.py`: `HE_LEVELS_CM1` (NIST He I n = 2), `KATO_3P2_3P1_MHZ`, `helium_parameters()` (G⁰, G¹, E_dir, ζ_S, ρ in cm⁻¹), `helium_parameters_ev()`, `fig_helium_2p_fan()`.
- `src/mercury.py`: `lande_g`, `components(B)`, `index(P, T)`, `dn_dp(T)`, `pressure_scan(P, B, e_mm, T, R)` (virtual recording), `predicted_peak_spacing_mmhg`, `bohr_from_peak_spacing(dP, B_gauss, T)` (the handout's analysis, mu_B/hc in cm⁻¹/G), `bohr_uncertainty`, `analyse_recording(dP_order, dP_pair, B_gauss, T, p_mean)` (whole lab analysis: e, mu_B/hc, uncertainty, deviation from CODATA), `analyse_pairs(pairs, B_gauss, T)` (several doublets: table, mean, scatter, standard error, deviation from CODATA), `fig_mercury_scan`. Script: `scripts/analyse_mercury_tp.py data/<file>.csv [B] [T] [sigma_B]`; data: `mercury_tp_measurements_TEMPLATE.csv` (to fill in) and `mercury_tp_example_SIMULATED.csv` (not a measurement).
- `src/torch_voigt.py`: `faddeeva(z)`, `voigt(x, sigma, gamma)` (NumPy or torch), `doppler_sigma_hz`, `lorentz_hwhm_hz`, `line_list(B, zeta, backend)`, `zeeman_spectrum(nu, B, T, P, ..., backend)`.
- `src/cavity.py`: `annihilation(n_max)`, `creation(n_max)`, `jaynes_cummings(n_max, w_c, w_a, g)` (basis: photon number ⊗ atom, index 0 excited), `excitation_number`, `excited_probability(...)` (exact time evolution), `fig_jaynes_cummings()`. Stand-alone, arbitrary units.
- `src/hyperfine_figures.py`: `fig_hyperfine()` (sodium 3S Breit-Rabi diagram and the hydrogen crossing splitting).

## `src/uncertainty.py`: how far can a number be trusted?

- `monte_carlo(model, inputs, n, seed)`: inputs are `{name: (mean, sigma)}`; draws each input from a normal distribution, runs `model(**values)`, returns mean, std and the error of the mean. Same seed gives the same answer.
- `linear_sensitivity(model, inputs)`: central-difference derivatives, each input's contribution to the final σ, and the total.
- `sigma_compare(a, sig_a, b, sig_b)`: z-score and a verdict (below 2 consistent, 2 to 3 borderline, above 3 disagreement).
- `hydrogen_crossing_gauss(lamb, fine)`: the 573 G crossing as a one-number model (root finding on the 2s and 2p blocks); `sodium_3p12_splitting_mhz(A)`: the 3P1/2 splitting from the full hyperfine matrix.
- Inputs are treated as independent. The Lamb-shift σ in the figure is the published 0.009 MHz (Lundeen and Pipkin); the H-alpha temperature σ in the budget is an assumption.
- `src/uncertainty_figures.py`: `fig_uncertainty()`.
- `src/uncertainty.py` also has `curve_band(curve_model, inputs, x, n, seed)`: Monte Carlo band (mean, std at every x) for a whole curve. `src/uncertainty_bands.py`: `fig_error_bands()` draws the hydrogen 2S level band (magnified 1000 times) and the sodium 3P1/2 level uncertainties.
- `src/budget.py`: the global table. `rows()` lists one `Row` per observable (system, model, inputs with σ, whether σ is measured or assumed, optional reference); `budget_table(n)` runs the Monte Carlo for all rows and adds sodium g_J against measurement; `fig_budget()` plots relative uncertainties. To add a system, append a `Row`.

## `src/stark.py`: hydrogen n = 2 in an electric field

- `dipole_operators()`: the x and z position operators of the n = 2 manifold in Bohr radii (from `hydrogen_spectrum.full_dipole`).
- `h_n2_stark(B, E, angle)`: the 8x8 Hamiltonian of `hydrogen.py` plus the electric term; angle 0 = E along B, 90 = across.
- `track_stark(...)`: level energies along a field sweep. `s_weights(...)`: how much 2s each eigenvector contains.
- `quenching_rate(B, E)` and `lifetime_s(rate)`: the 2P admixture times the 2P decay rate, and the resulting lifetime.
- `perturbative_s_shift_mhz(E)`: the textbook second-order shift, used as an independent check.
- `anticrossing_gap_mhz(E)`: the smallest gap near 573 G, and where it occurs.
- `stark_figures.py`: `fig_stark_fan()` (levels and lifetime against E) and `fig_anticrossing()` (the opening gap).

## `src/hydrogen_spectrum.py`: Extension C, hydrogen H-alpha

Hydrogen n = 3 to n = 2 gives a real spectrum (the red Balmer line, 656 nm). The Hamiltonian of a shell is block-diagonal in l.

- `radial_function(n, l, r)`: hydrogen radial wavefunction (generalised Laguerre polynomials), in Bohr radii.
- `radial_integral(n_up, l_up, n_low, l_low)`: the radial part of the dipole matrix element, by numerical integration (cached).
- `einstein_a(...)`: spontaneous decay rate in 1/s; used as a check against literature values.
- `dirac_energy_cm1(n, j)` and `lamb_shift_cm1(n)`: fine structure and the (scaled) Lamb shift.
- `zero_field_block_ev(n, l)`: chooses a constant and a spin-orbit constant so each l block has the two Dirac energies.
- `manifold_hamiltonian(n, b_tesla)`: all states of shell n, block by block, with the Zeeman term.
- `full_dipole(n_up, n_low)`: angular part (Wigner-Eckart, from `transitions.py`) times radial integral, for q = -1, 0, +1.
- `balmer_components(b_tesla, n_up=3, n_low=2)`: the line list, same columns as the sodium one, so every filter in `optics/` works on it.
  Intensity columns, which are three different things: `line_strength` is the squared dipole matrix element in Bohr radii squared (wave functions only, no frequency); `einstein_a` is the physical spontaneous-emission rate of that one component in 1/s, A = (4/3) alpha^3 w^3 |d|^2, with w the component's own transition energy; `rate` is the relative photon-emission weight used by the optics filters, line_strength times (w/w0)^3 (differs from line_strength by ~1e-4 across a multiplet). The sum of `einstein_a` over all components divided by the 18 upper states equals the degeneracy-weighted mean of `einstein_a()` over 3s, 3p, 3d.
- `doppler_fwhm_pm(...)`: thermal line width, about 8.1 pm at 300 K.

**`optics/hydrogen_figures.py`**: `fig_halpha_regimes` (three field strengths times three filters, with Doppler blur) and `fig_halpha_fan` (components against field) and `animate_halpha_field_ramp` (GIF: B rises, three filter views update together).

## `src/animation.py`: plots and animation

- **`plot_energy_levels(...)`** static energy-versus-field diagram built on `track_levels`.
- **`animate_splitting(...)`** the same curves drawn progressively with `FuncAnimation`. Saved as a GIF with `save_path`.
- **`plot_stick_spectrum(...)`** a stick spectrum (position = shift, height = intensity, colour = polarisation).
- **`plot_n2_levels(...)`** hydrogen n = 2 levels versus field, coloured by term, with optional dotted lines at chosen fields.
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
| `test_spectrometer.py` | Zero-field D-line wavelengths; 4 + 6 components; ~27 pm/T shift; D1 components equal and D2 in the ratio 4 : 3 : 1 across B; no π lines along B; polariser ∥ B keeps only π; circular analysers separate σ+ from σ−; the prism cannot resolve Zeeman components at 1 T. |
| `test_instruments.py` | Grating and Fabry-Perot formulas; ladder ordering; required resolving power ∝ 1/B; field-for-resolving-power is consistent; the instrument-ladder figure builds. |
| `test_hydrogen.py` | Zero-field scheme; Dirac cross-check; labels; Landé slopes for 2S₁/₂, 2P₁/₂, 2P₃/₂; B never mixes 2s with 2p; the 2S–2P crossing near 570 G; Paschen-Back slopes. |
| `test_two_electron.py` | Algebra and Hermiticity; degeneracies 5, 3, 3, 1; Landé interval rule 2 : 1; g = 1.5 / 1.5 / 1; admixture equals perturbation theory; zero without spin-orbit; fit recovers known parameters; calcium fit consistent; rate scaling. |
| `test_literature.py` | Measured g_J of the P states within 1 sigma, S state off by the bound-state size; reference table matches its own formulas; model within 1 % of paper numerics and 0.5 % of its formulas; spacings scale with B; mu_B value. |
| `test_hyperfine.py` | Hermitian; Na 3S splitting 1771.626 MHz with 3 + 5 degeneracy; stretched state matches its exact formula; 3P1/2 splitting 2A; H 2S/2P splittings; proton g_I; F_z conserved; the 573 G crossing splits into four with mean 570 to 573 G. |
| `test_uncertainty.py` | Sampling reproduces the exact σ of a linear model; derivatives match algebra; seed reproducibility; z-score verdicts; crossing 573.4 G with slope about 0.55 G/MHz; sampled σ of the sodium splitting equals 2σ_A. |
| `test_budget.py` | Every system present; every σ labelled measured/assumed; sodium hyperfine row matches exact σ; no calcium hyperfine row; g_J(3P) consistent and g_J(3S) flagged. |
| `test_hyperfine_lines.py` | 6 D2 and 4 D1 zero-field lines; Racah strength fractions 1/20, 1/4, 7/10 and 1/6, 5/12, 5/12; D2:D1 = 2; line spacings 3A and 1771.626 MHz; D2 total strength field independent; Doppler width scales as √T; broadened area equals strength. |
| `test_line_profiles.py` | Na D2 natural width 9.80 MHz; unit area; Gaussian and Lorentzian limits; approximate width within 0.5 % of exact; Voigt spectrum area equals strength; Gaussian limit equals the Doppler-only spectrum. |
| `test_corrections.py` | Diamagnetic 1s = B²/4 hartree (atomic units); ⟨r²⟩ formula equals numerical integral for five states; angular average over m is 2/3; shift scales as B²; below 1 % of Zeeman at 60 T; isotope shift sign and size (about 179 pm, D bluer); Zeeman scale equal for H and D; regression for the Doppler-width call. |
| `test_calcium_lines.py` | Normalised dipole state; sum rule 1 per polarisation; only J = 1 radiates at B = 0; 3P₁ strength = 3ε², 1P₁ = 3(1−ε²); 3P₀ ∝ B² and π only; 657.46 nm; σ± split = 2 g μB B |
| `test_helium.py` | Hermitian; exchange = ±G (2G gap, 3:9 degeneracy); equals the calcium Δ up to a constant; spin-spin zero in singlets, traceless, 1 : −5 : 10; Landé 2 : 1 without it; closed form = diagonalisation; interval fit round-trips; 1s2s g = g_s; F⁰(1s,1s) = 5Z/8 |
| `test_helium_data.py` | NIST and Kato intervals agree (1.5e-4 cm⁻¹); calibrated model reproduces all n = 2 levels; G⁰ = 3210.73, G¹ = 1023.99 cm⁻¹; triplet inverted, Landé ratio ≈ −0.08; hydrogenic G too large by 3.7 and 2.5 |
| `test_mercury.py` | Landé 2 and 3/2 (and the engine's 3P₂); nine components, δ = −2…2; strengths sum rule 5/3 per M_u and symmetric; 6 σ + 3 π; μB/hc equals CODATA; the 0.0089 nm/T statement of the handout; index law; ΔP = 32.45 mmHg at 2340 G; simulated recording recovers μB/hc within 2e-3 |
| `test_torch_voigt.py` | Faddeeva and Voigt equal scipy (NumPy); unit area; widths ∝ √T and + P; line list equals zeeman_lines; torch gradcheck and B, T, P finite differences (skipped without torch) |
| `test_cavity.py` | [a, a†] = 1 except the last state; number operator; excitation number conserved; resonant levels (N−½)w_c ± g√N; vacuum Rabi cos²(g t); detuned Rabi formula. |
| `test_uncertainty_bands.py` | Band of a linear model matches algebra; the hydrogen 2S band equals the Lamb-shift σ; sodium level uncertainty is below the natural width and at most about 2σ_A. |
| `test_stark.py` | Hermitian operators with -3 a0 / 3/sqrt2 a0; E = 0 reproduces hydrogen.py; isotropy at B = 0; 2S Stark shift vs perturbation theory; anticrossing near 573 G; gap linear in E and equal to 2 sqrt3 e a0 E; parallel E leaves the crossing; quenching rate and lifetime limits. |
| `test_tracking.py` | Crossing on a grid point, tangential touch, plateau of zeros, closely spaced crossings, refinement vs linear interpolation, grid independence, and a true eigenvalue degeneracy at the refined field; One-to-one (Hungarian) assignment where per-row argmax double-claims a level; genuine crossing followed straight through; avoided crossing gives the adiabatic branches; reversed sweep; near-degenerate pair; orthonormal columns in a real sweep; warning on a too-coarse step and the returned overlap confidence. |
| `test_halpha_instruments.py` | R needed scales as 1/B; inverse function; ladder ordering; Doppler floor about 81 000; prism too weak but big grating enough at 1 T; prism deflection scales with B. |
| `test_grotrian.py` | Every arrow satisfies M_up − M_low = q at five fields; polarisation matches q; M values are the expected half-integers; 10 arrows, 10 strong at 0.1 T and 6 strong at 60 T; no duplicate arrows. |
| `test_sodium_paschen_back.py` | Mean sigma shift = normal Zeeman 16.2 pm/T at 1, 10, 60, 120 T; 10 components; strong lines dominate at 60 T. |
| `test_fabry_perot.py` | Finesse formula; Airy peak and gap; one ring at zero field; six resolved peaks at 0.3 T; polarisers leave 4 sigma or 2 pi peaks; heights near 4 : 3 : 1. |
| `test_hydrogen_spectrum.py` (incl. line strength vs omega^3 rate, component sums vs `einstein_a`) | Radial functions normalised; exact Lyman-alpha integral; decay rates vs literature; fine structure = alpha^2 Ry/16; Lamb shift; level counts; equal total emission per polarisation; strong-field sigma shift = 20.11 pm/T; polariser keeps only pi; Doppler width. |

Run them all with `python -m pytest -q`.

## `scripts/`

- **`make_figures.py`** regenerates every figure in `docs/figures/` (add `--gif` for the animation).
- **`make_optics_figures.py`** renders the virtual-spectrometer figures and, with `--gif`, its six animations.
- **`run_all.py`** one command for everything (tests, figures, notebooks, PDF). Options: `--fast` (no GIFs), `--no-pdf`.
- **`make_notebooks.py`** writes the seven notebooks from their source text.
- **`execute_notebooks.py`** runs all notebook cells and saves the printed output and plots inside the `.ipynb` files.

## `notebooks/`

One lab per level, each with theory in markdown, code, and saved output.
1. Normal Zeeman and the Lorentz triplet · 2. Spin-orbit, Landé g and ζ from sodium ·
3. Weak field to Paschen-Back, with tracked levels · 4. Selection rules, line intensities and fitting ·
5. Hydrogen n = 2: Lamb shift, level crossings, and why B never mixes 2s with 2p ·
6. The virtual spectrometer: colours, prism, polarisation filters, and which instrument can resolve the Zeeman pattern ·
7. Hydrogen H-alpha: a real spectrum with polarisation filters, from weak field to Paschen-Back.

## `docs/` and `.github/`

- `docs/main.tex` includes `symbols.tex` (all symbols), one `.tex` file per level, one `ext_*.tex` per extension (hydrogen, optics, H-alpha, Stark, two-electron, literature, hyperfine, Voigt, corrections, cavity, uncertainty), a validation table, and `ext_sources.tex` (every input with its source, figure provenance and the bibliography). Build with `cd docs && latexmk -pdf main.tex`.
- `.github/workflows/latex.yml` runs the tests, executes the notebooks and builds the PDF on every push.

---

## How to add a new system (the method)

1. Write the Hamiltonian pieces in `hamiltonians.py` (or a new file).
2. Diagonalise with `solvers.py` and plot with `animation.py`.
3. Find a measured number, add it to `data/experimental_reference.csv`, and write a test that compares it to your output.
4. Only when the test passes, move on to the next level.
