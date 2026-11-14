# Stage 8 — Uncertainty-Propagation Audit (read-only)

Scope: Quantum-Zeeman-Engine as in the last delivered zip. **No source, test, notebook, figure or document was modified.** All numbers below were produced by a scratch script outside the repo.
Legend: **V** = VERIFIED (inspected and established), **PV** = PARTIALLY VERIFIED, **U** = UNVERIFIED (needs external source), **NI** = NOT IMPLEMENTED. "Inspected" = I read the code or ran it; "inferred" = my reasoning, not shown by a run.

---
## 1. Current uncertainty infrastructure

| Item | Where | Status |
|---|---|---|
| Monte Carlo propagation, independent normal inputs, `sem = std/sqrt(n)` | `src/uncertainty.py::monte_carlo` | V |
| Central-difference first-order propagation, diagonal covariance only (`sigma^2 = sum (df/dx_i sigma_i)^2`) | `uncertainty.py::linear_sensitivity` | V |
| z-score with fixed thresholds 2 / 3 | `uncertainty.py::sigma_compare` | V (conventions, not confidence levels) |
| Whole-curve bands by sampling | `uncertainty.py::curve_band`, `uncertainty_bands.py` | V |
| Global budget table (5 rows + 3 g_J rows), each row labelled `measured`/`assumed` | `src/budget.py::rows, budget_table` | V, but see finding F4 (stale calcium rows) |
| Jacobian + full covariance propagation with correlation matrix | `two_electron.py::calcium_fit_sensitivity` only | V (calcium only, assumed sigmas) |
| Covariance from a least-squares fit | — | **NI** (`fitting.py::fit_parameters` returns `rms_residual_ev` only; `sol.jac` is computed by SciPy but discarded) |
| Correlated inputs in MC / `linear_sensitivity` | — | **NI** (module docstring states "no correlations") |
| Central input register (value, unit, sigma, type, source) | — | **NI**: values are scattered over `hyperfine.py`, `hydrogen.py`, `helium_data.py`, `two_electron.py`, `literature.py`, `data/experimental_reference.csv` |
| Model-uncertainty ledger | — | **NI** (limits are prose in docstrings and `AUDIT_LOG.md`) |

Inputs to `monte_carlo`/`linear_sensitivity` are scalar (mean, sigma) pairs hard-coded in `budget.py`, `uncertainty_bands.py`, `uncertainty_figures.py`; they are not read from the modules that hold the physical constants (F2).

---
## 2. Input data and reference audit

| Input | Value | Source (as stated in code) | Source gives sigma? / kind | Code uses sigma? |
|---|---|---|---|---|
| CODATA constants (mu_B, g_s, hbar, a0 …) | scipy.constants | CODATA via SciPy | CODATA provides uncertainties (`physical_constants[...][2]`) — recommended | **No**: `constants.py` uses `[0]` only. CODATA edition of the installed SciPy not recorded (U) |
| H Lamb shift 2S–2P1/2 | 1057.845 MHz | Lundeen & Pipkin, quoted in arXiv:0809.2985 | 0.009 MHz, measured | Only as a hard-coded 0.009 in `budget.py`/`uncertainty_bands.py`; `hydrogen.py` holds the central value only |
| H 2P3/2–2P1/2 | 10969.13 MHz | Baird et al., PRA 5, 564 (1972) | 0.10 MHz, measured | **No** (comment only) |
| H 2S hfs | 177.55683887 MHz | measured (docstring) | 0.85e-6 MHz | `H_HFS_2S_SIGMA_MHZ` defined, used in one test only |
| H 2P1/2 hfs | 59.2212 MHz | theory quoted in arXiv:1104.4421 (measured there 60(2)) | theory value; sigma of theory not recorded | No |
| H 2P3/2 hfs | Fermi/60 | leading-order theory, not checked against a source (stated in code) | none | No — **U** |
| Na A(3S), A(3P1/2), A(3P3/2) | 885.8130644, 94.44, 18.534 MHz | Steck (web page), A(3P3/2) later compared with Das et al. 2006 18.530(3) | Steck quotes sigma (0.13 for A(3P1/2), in budget only); others none in code | Only A(3P1/2), duplicated at 3 sites |
| Na B(3P3/2) | 2.724 MHz | Yei et al. 1993, 2.724(30) | 0.030 measured | No (comment only); Das 2006 gives 2.721(8) — two sources not combined |
| Na g_J (3S, 3P1/2, 3P3/2) | 2.0022960(7), 0.66581(12), 1.3342(2) | Steck Table 6, "quoted via a web-page summary: check against the PDF" (`literature.py`) | measured sigma | Yes (`g_factor_comparison`), but source text unchecked — **U** |
| He levels | 6 NIST (M02) values | NIST Handbook | no sigma recorded in code | No |
| He 3P2–3P1 | 2291.176590 MHz | Kato, Skinner, Hessels arXiv:1807.07923 | 25 Hz measured | No (comment) |
| Ca levels | 3P0, 3P1, 3P2, 1P1 | NIST Handbook SC85 | **none** (code says so) | `budget.py` inserts 0.001 cm^-1 "assumed" = table rounding, not an uncertainty |
| Ca 3P1 measured freq | 455986240494135.8 Hz | `CA_3P1_MEASURED_HZ` | sigma not stored | — U |
| 1P1 Einstein A | 2.20e8 /s | primary measurement not found (AUDIT_LOG) | none | No — U |
| Wiese–Fuhr A values | tests only | Wiese & Fuhr 2009 Table 6 | graded tables, grade not stored | No |
| N_AIR | 1.000277 constant | convention | wavelength dependence ignored (inferred) | — |
| H-alpha temperature | 300 ± 10 K | "assumption about the lab" (`budget.py`) | assumed | Yes, labelled `assumed` |

**F2 (V):** only five sigmas reach any propagation (Lamb, A(3P1/2), g_J ×3, T) and each is typed by hand at the call site. No sigma is invented by the code except the two labelled `assumed` (Ca 1e-3 cm^-1, T ± 10 K).

---
## 3. Fitted parameters and covariance

| Model | Fitted | Fixed | Data | dof | Covariance | Status |
|---|---|---|---|---|---|---|
| `fitting.py::fit_parameters` (zeta, optional g_s) | zeta [, g_s] | rest | **synthetic** in all tests/notebooks (`tests/test_fitting.py`, noise 1 micro-eV) | n_B×levels − p (large) | **NI** | Recovery tests = implementation checks, not validation; no real-data fit exists in the repo |
| Hydrogen n=2 | none (L, F from literature) | all | — | — | — | not a fit |
| Helium `helium_data.py::helium_parameters` | 6 (G0, G1, two E_dir, zeta_S, rho) | zeta_A = 0 / bounded | 6 NIST levels + Kato interval | **0** | none | Calibration; module docstring says so (V) |
| Calcium `calcium_four_parameter_fit` | 4 (zeta_S, zeta_A, Delta, E_t) | — | 4 levels | **0** | `calcium_fit_sensitivity` (Jacobian, covariance, correlation) with assumed sigmas | Exact closure = calibration; the covariance is valid only as a scenario because sigma is assumed (V) |
| Calcium `fit_to_levels` (3 params / 4 levels) | zeta, Delta, E_t | zeta_A = 0 | same | 1 | none | **Does not reproduce the data**: residuals (3P0, 3P1, 3P2, 1P1) = +0.060, −0.089, +0.029, +7e-6 cm^-1 (V, run) |

**Statistical meaning of parameter uncertainties:** none exist for the fits except calcium's assumed-sigma scenario (zeta_A ± 0.056 cm^-1 of 46.75 for sigma = 0.001 cm^-1 on every level; eps² = 6.13e-5 ± 1.5e-7, V run). With zero dof there is no chi², no goodness of fit and no test of the model; the propagated errors are the inverse-map Jacobian times the *assumed* level sigma. **Helium:** no propagation exists (NI). The He "consistency check" and Ca lifetime comparison use data not in the fit but carry the open-reference caveats in `AUDIT_LOG.md`.

---
## 4. Numerical uncertainty

| Item | What exists | Status |
|---|---|---|
| Eigenvalue precision (`eigvalsh`, float64) | not measured; mpmath is not installed so I could not run a high-precision comparison. Inferred scale: machine epsilon × ‖H‖ ≈ 1e-12 MHz for h_n2 (inferred, not run) | U |
| Eigenvector tracking | Hungarian assignment, `min_overlap` warning, `return_confidence`; degenerate subspaces documented | V (controlled by tolerance, not quantified as an error) |
| Field-grid resolution | grid-independence tests for crossings (`tests/test_tracking.py`); **two crossings inside one grid interval are still missed** (documented) | PV |
| Crossing refinement | `refine_crossings_from_builder` xtol 1e-12 T (floor ≈ 1e-13 T); `hydrogen_crossing_gauss` uses `brentq xtol=1e-9 T` = 1e-5 G and a hard-coded bracket 0.050–0.065 T and **index-based** level selection (`eigvalsh(...)[1]`) rather than tracking | V (1e-5 G ≪ sigma 0.005 G) |
| Radial integrals | `quad(limit=400)` on [0, 70 n²]; R(2p→1s) = 1.2902662019598634 vs analytic 1.2902662019598632 (V, run). Truncation at 70 n² and `quad` error estimates not examined for n ≥ 3 | PV (one case) |
| Einstein A, line strength | closed-form tests (2p→1s); Wiese–Fuhr comparison residual 1.1e-4 attributed to relativistic corrections (not computed) | PV |
| MC sampling noise | `sem` returned; I saw std of the crossing vary 4.57e-3 / 4.76e-3 / 4.76e-3 G for seeds 1/2/0 at n = 200 versus linear 4.94e-3 (V, run) — consistent with 1/sqrt(2n) ≈ 5 %; the n = 40 used in tests/figures has ≈ 11 % | V |
| Finite-difference step | `linear_sensitivity` rel_step 1e-4 fixed; `calcium_fit_sensitivity` h = 1e-4 cm^-1 fixed; neither checked for step independence | U |
| Numerical precision vs physical accuracy | The documentation does say the model, not the inputs, limits agreement for H and Na | PV |

---
## 5. Propagation methodology

For y = f(x), σ_y² ≈ J C Jᵀ with J_ij = ∂f_i/∂x_j and C the input covariance.

* **Valid** when f is smooth on the scale of σ: the neglected second-order term ≈ ½|f''|σ² must be ≪ |f'|σ. Checked for the hydrogen crossing: f'' = 1.2e-5 G/MHz², ratio at 1σ = 2e-7 → linear propagation is exact for practical purposes (V, run). Calcium eps² = (a − l1)/(l2 − l1) is smooth, MC (7.785e-5 ± 1.46e-9) matches linear for the budget's definition (V, run).
* **Needs MC / nonlinear treatment** near level crossings or anticrossings (non-analytic gap), near degenerate-eigenvector labelling changes, for observables that are minima over a grid (`anticrossing_gap_mhz`), for the V² ≥ 0 constraint in `calcium_four_parameter_fit` (the `ValueError` branch makes the map non-smooth at the boundary), and when σ is not small compared with the scale of f.
* **Correlated inputs:** a correlated covariance is needed when two inputs share a source (Das 2006 A and B; Yei and Das for B; H hfs 2P intervals) — currently NI; `curve_band`/`monte_carlo` draw independent normals only.
* **Fit parameters:** after a real fit, C_par = s² (JᵀJ)⁻¹ with s² = χ²/dof (or measurement-sigma weighted); exact fits (dof 0) have no s². NI.
* **Sensitivity scenarios** (e.g. the ±0.001 cm^-1 Ca scenario) are legitimate as "what-if" tables but must not be called standard errors.

Sensitivities computed (run): crossing field ∂B/∂L = +0.5485 G/MHz, ∂B/∂F = −6.2e-4 G/MHz, ∂B/∂g_s = −133 G per unit g_s. With published sigmas: Lamb 0.0049 G, fine-structure 6e-5 G, so the budget row (Lamb only) already captures the input uncertainty to 1 %.

---
## 6. Observable-by-observable assessment

| Observable | Dominant uncertainty sources | Enough information to propagate? |
|---|---|---|
| Energy levels (H, Na) | Lamb, fine, hfs inputs (sigma known for some); model terms | H: yes for L, F, hfs 2S; Na: A(3P1/2) only |
| Transition freq / wavelength | level inputs; air/vacuum convention constant | partly; N_AIR dispersion unquantified |
| Line strengths | model (basis truncation), numerical quad | no input sigma; numerical error not assessed |
| Einstein A, lifetimes | radial integral (numerical, accurate to ~1e-16 for 2p→1s), reduced-mass, relativistic corrections 1e-4 | no reference sigma for compared A values (grade of Wiese–Fuhr table not stored) |
| Crossing fields | H: Lamb (0.005 G) — tiny; **model: hyperfine moves the crossings by ~1.8 G (mean of four 571.6 G vs 573.4 G) and spreads them ~3 G**; diamagnetic 2e-4 G | statistical part yes; reference "about 570 G" has no sigma, so no z-score possible |
| Avoided-crossing gaps | matrix element (exact in model), E-field calibration (not an input here), 2.3 % 2P3/2 admixture is physics, hyperfine absent | no experimental sigma available |
| Hyperfine intervals | A, B constants (sigma known from Steck/Das/Yei but not wired in) | yes for Na if inputs wired and Yei/Das correlation decision made |
| Stark shift | E-field (user input), matrix element, n≥3 neglected (2e-6 MHz at 10 V/cm) | yes numerically; no spectroscopic reference yet |
| Diamagnetic shift | formula exact within n=2; reduced-mass factor (1.00163) unresolved; n=3 s–d coupling not diagonal | no sigma required beyond B |

---
## 7. Model uncertainty and limitations (not to be given statistical error bars)

1. **Missing configuration mixing** (He beyond 1s2p/1s2s, Ca perturbers: 4s5p ¹P/³P J=1 present only in a table) — systematic, unknown sign and size.
2. **Hyperfine approximations:** H 2P3/2 hfs leading-order; hydrogen J-mixing ≤ 7 kHz (±0.1 G) deferred; ⁴³Ca not modelled; Na quadrupole included only for 3P3/2.
3. **Stark truncated to n=2** and no hyperfine: validity range stated (n≥3 quadratic term negligible below ~1 kV/cm).
4. **Diamagnetic:** not in any Hamiltonian; n=2 exactly diagonal; n≥3 requires degenerate treatment (253 MHz coupling at 20 T for 3s–3d).
5. **g_J(3S1/2) 33σ gap** (`literature.py`, `budget.py::g_factor_rows`): model deficiency (free-electron g_s). The attribution to bound-state corrections is stated in `ext_uncertainty.tex` but no corrected value was computed or sourced (PV).
6. **Lamb shift for n=3** scaled as 1/n³ ("a few percent", unquantified).

Why these are not statistical: they are deterministic omissions with a definite but unevaluated sign, they are not drawn from a distribution, they are correlated across observables, and averaging does not reduce them. Representing them as σ would let a user combine them in quadrature with Lamb-shift-level uncertainties (0.005 G) and imply a coverage probability that no sampling justifies. The correct treatment is a separate "model-sensitivity" column or scenario (e.g. with/without hyperfine) with no confidence-level language.

Four-way split currently possible: (1) input/experimental: H, Na, g_J values with σ; (2) parameter: Ca only as a scenario; (3) numerical: partial; (4) model: prose only; (5) sensitivity scenarios: Ca 1e-3 cm^-1, T ± 10 K (both correctly labelled `assumed`).

---
## 8. Existing tests and missing tests

Existing (all categories are implementation or analytic-consistency, none spectroscopic): `tests/test_uncertainty.py` (6: linear MC, algebra, seed, z verdicts, crossing slope 0.5–0.6 G/MHz, Na 2σ_A), `test_uncertainty_bands.py` (3), `test_budget.py` (5: rows present, sigma source labels, Na row 188.88 ± 0.26, no Ca hyperfine row, g_J verdicts), `test_fitting.py` (3 synthetic-recovery), `test_two_electron.py` (calibration/calcium), `test_tracking.py` (grid independence). `pytest.ini` has markers `implementation`, `calibration`, `spectroscopic`.

**Findings on conventions and wording**
* **F4 (V, run) — stale calcium rows in the published budget.** `budget.py::_ca_admixture` calls `fit_to_levels` (three-parameter fit, zeta_A = 0, residuals up to 0.089 cm^-1) and returns eps² = 7.785e-5 with sigma 1.3e-9. The audited closed form `calcium_four_parameter_fit` gives eps² = 6.134e-5 (= displacement/separation), and its propagation with the same assumed sigma gives 1.5e-7 (≈ 110× larger). `docs/ext_uncertainty.tex` table repeats 7.8e-5 ± 1e-9. The two values differ by 27 %; the budget row is therefore inconsistent with the calcium audit stage. (Not corrected — awaiting approval.)
* The Na g_J rows are labelled `sigma_source = measured` with model sigma 0; z is then purely (model − measurement)/σ_meas: a model-error indicator, not a consistent statistical test.
* `sigma_compare` thresholds 2/3 are conventions (normal assumption) and the module speaks of "real disagreement" at z > 3.
* `ext_uncertainty.tex`: text says 188.87 ± 0.26 MHz, table 188.88 (rounding of the run).
* The same sigma (94.44 ± 0.13) is typed at four sites (budget, bands, figures, tests) instead of one record.
* "assumed σ = table precision" (Ca 0.001 cm^-1) is a rounding interval; treating it as σ of a normal draw is a scenario, which the budget labels `assumed` and the text states.
* Terms "accuracy/validation": `validation.py` docstring says "if the numerical result reproduces the measured number, the model is correct" — an overstatement (PV); repo-level markers (`calibration` vs `spectroscopic`) are the better wording.

**Missing tests (proposals only):** budget consistency with the closed-form calcium fit; MC-vs-linear agreement bound for the crossing with both L and F; step-size independence of finite differences; a fit-covariance test against a known analytic linear case; `linear` vs `monte_carlo` agreement near the V² = 0 boundary of the calcium map; convergence of `quad` against tighter limits for n ≥ 3.

---
## 9. Recommended implementation priorities (after approval)

1. Reconcile `budget.py` calcium rows with `calcium_four_parameter_fit` and rebuild table/figure (correctness).
2. One input register: (value, unit, σ or None, kind ∈ {measured, theory, recommended, assumed}, source, used-by) — no invented σ; wire Lamb, F, A, B, g_J through it.
3. `fit_parameters`: return Jacobian-based covariance only when measurement σ are supplied; flag dof = 0.
4. Crossing-field propagation with L, F (and hfs constants) including documented correlations; keep hyperfine as a scenario column.
5. Numerical-convergence report (eigenvalue precision with an independent high-precision tool, quad limits, grid, finite-difference step).
6. Observable-level uncertainty for hyperfine intervals, then A coefficients (only with graded reference σ).
7. Separate "model sensitivity" ledger (scenarios; no confidence language).

## 10. Unresolved questions requiring external references

* CODATA edition used by the installed SciPy and constants' uncertainties.
* Primary sources: Steck Table 6 g_J (checked only via a web summary), Na A(3S, 3P) sigmas, Das 2006/2008 values vs Yei 1993 — independence/correlation.
* Theory uncertainty of H 2P1/2 and 2P3/2 hfs; published n=3 hfs.
* NIST uncertainties for He and Ca levels (code says none listed); primary measurement for A(1P1) = 2.20e8 /s; Ca 410 Hz value.
* Quantitative bound-state correction to g_J(3S1/2) for sodium.
* A published hydrogen n=2 anticrossing/Robiscoe quantity with hyperfine for Stark validation; published strong-field hydrogen diamagnetic tables; two-body reduction of the diamagnetic term.
* Wiese–Fuhr grade (uncertainty class) for each compared A value.

## Proposed roadmap (scientific importance × feasibility)

| Order | Step | Why |
|---|---|---|
| 1 | Fix stale calcium budget rows (F4) | published numbers currently inconsistent; small change |
| 2 | Input register with typed uncertainties | prerequisite for everything below; no new physics |
| 3 | Hydrogen crossing: multi-input propagation + hyperfine scenario column | most defensible observable (σ_stat 0.005 G vs model ~2–3 G) |
| 4 | Fit covariance infrastructure | needed before any real-data fit (mercury, lab data) |
| 5 | Numerical convergence report | separates precision from accuracy |
| 6 | Hyperfine-interval propagation (Na) with correlation decision | needs source clarification |
| 7 | Einstein-A/lifetime uncertainties | needs graded reference σ |
| 8 | Stark / diamagnetic uncertainty | blocked on validation references |

Awaiting reviewer approval before any change.
