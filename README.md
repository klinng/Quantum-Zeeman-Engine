# Quantum Zeeman Engine

Zeeman physics built **level by level**, each level with a derivation (LaTeX), a numerical model
(Python), a plot/animation, and a check against experimental values. If the computed result
reproduces the measurement, the model underneath is correct.

| Level | Physics | Code | Report |
|---|---|---|---|
| 1 | Normal Zeeman, ΔE = μ_B B m_l | `hamiltonians.h_zeeman` (ζ = 0) | `docs/level_1_normal.tex` |
| 2 | Spin-orbit, Landé g_J | `hamiltonians.landé_g`, `h_spin_orbit` | `docs/level_2_anomalous.tex` |
| 3 | Paschen-Back, level repulsion | `solvers.track_levels`, `animation.py` | `docs/level_3_paschen_back.tex` |
| 4 | Wigner-Eckart, tensors, ζ fitting | `torch_engine.py` (optional) | `docs/level_4_tensors.tex` (planned) |

## Hamiltonian

H = H₀ + ζ **L·S** + μ_B B (g_l L_z + g_s S_z), built in the basis |l, m_l⟩ ⊗ |½, m_s⟩ and
diagonalised with `numpy.linalg.eigh`. ζ is set from the measured sodium 3p fine-structure
splitting (17.196 cm⁻¹); everything else is a prediction.

## Quick start (VSCode)

```bash
pip install -r requirements.txt
python -m pytest -q                  # 26 tests
python scripts/make_figures.py --gif # figures + animation into docs/figures/
cd docs && latexmk -pdf main.tex     # build the PDF report
```

## Layout

- `src/` physics code · `tests/` Hermiticity, Landé g, Paschen-Back limit, sodium check
- `data/experimental_reference.csv` measured values used for validation (verify against NIST ASD)
- `notebooks/` interactive labs · `docs/` LaTeX report · `scripts/make_figures.py`

## Status

Levels 1 to 3 are implemented and tested. Level 4 is planned.
