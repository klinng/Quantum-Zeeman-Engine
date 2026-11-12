# Audit log

Each stage is: finding, change, test, remaining limit, and what still has to be checked against a primary source.

| Stage | Finding | Change | Remaining |
|---|---|---|---|
| 1. Eigenstate tracking | per-row argmax can assign one new eigenvector to two old levels; silent energy-sort fallback | global one-to-one assignment (Hungarian); RuntimeWarning when the best matched overlap < 0.5; `return_confidence` | inside an exactly degenerate subspace only energies are defined, not vectors |
| 2. Hydrogen rates | the `rate` column was a squared dipole matrix element (a line strength) | columns `line_strength`, `einstein_a` (omega^3, per component), `rate` (relative photon weight) | none known |
| air/vacuum test | test mixed air wavelength with a vacuum constant (8e-4 = (N_AIR-1)^3 * 3) | test inverts the code's own convention, tolerance 1e-9 | N_AIR is one constant (1.000277), not dispersive |
| 4. Reduced mass | energies used reduced-mass Rydberg, radial integrals the infinite-mass length: A scaled as mu^3 instead of mu | `A_MU_OVER_A0`; A now = A_inf * mu/m_e (0.054 %); compared with Wiese and Fuhr 2009 Table 6 (<= 1.1e-4) and the closed form (2/3)^8 alpha^5 m_e c^2/hbar | relativistic terms in the tabulated A |
| 5. Crossings | sign test misses grid-point zeros; plateaus reported as a crossing; refinement missed grid-point roots | `find_crossings` (exact zeros, `include_touching`), `find_zero_intervals`, `refine_crossings_from_builder` (re-diagonalise inside the bracket, floor ~1e-13 T) | two crossings inside one grid interval are still missed; a degenerate pair at the reported field does not prove all crossings were found |
| Helium zeta_A and labels | zeta_A = 0; `classify` rounded <J^2>, misleading above ~0.05 T | `classify` returns M_J, J-weights, dominant J and purity; `classify_tracked` gives zero-field parentage; selection rules and the second-order admixture are tested; report paragraph on the size of zeta_A | **to confirm in the primary papers:** the 4.755 / 3.757 MHz second-order mixing shifts (arXiv:2510.17495 Table III and its footnote, quoting Drake 1993 and Pachucki 2006) and the ~177 1/s 2^3P_1 -> 1^1S_0 rate (Drake and Morton 2007; PRL 100, 023001 (2008)). They were read from page summaries, not from the tables themselves. |
