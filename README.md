# Quantum Zeeman Engine

[![Tests and PDF](https://github.com/klinng/Quantum-Zeeman-Engine/actions/workflows/latex.yml/badge.svg)](https://github.com/klinng/Quantum-Zeeman-Engine/actions/workflows/latex.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Take one big physics problem, split it into levels, and make every level checkable:**
a LaTeX derivation, a Python model, a plot or animation, and a comparison with measured values.
If the computed number reproduces the experiment, the model underneath is correct.

We use the **Zeeman effect** (how a magnetic field splits atomic energy levels) as the first
case study. The same method works for any problem you can build up in layers.

<p align="center">
  <img src="docs/figures/p_orbital_levels.png" width="500" alt="Energy levels of a p electron versus magnetic field, from weak field to Paschen-Back">
</p>

*Energy levels of a p electron (sodium 3p spin-orbit constant) from the weak-field regime to
Paschen-Back. Levels with different m_j cross; levels with the same m_j repel.*

<p align="center">
  <img src="docs/figures/na_d_lines_zeeman.png" width="760" alt="Computed Zeeman components of the sodium D1 and D2 lines with polarisation and intensity">
</p>

*Predicted Zeeman pattern of the sodium D lines at 1 T: 4 components for D1, 6 for D2, with
polarisation (colour) and relative intensity (height). Nothing here is typed in by hand; it comes
from diagonalising the Hamiltonian and applying the dipole operator.*

## Why decompose a problem?

1. **Understand** the phenomenon, the math and the physics behind it, one layer at a time.
2. **Verify** each layer against known results before adding the next, so errors can't hide.
3. **Compute, plot, animate** and run numerical experiments on a model you trust.

## The levels

| Level | Physics | Key result | Code | Report |
|---|---|---|---|---|
| 1 | Normal Zeeman (orbital only) | ΔE = μ_B B m_l, Lorentz triplet | `hamiltonians.h_zeeman` | `docs/level_1_normal.tex` |
| 2 | Spin-orbit coupling, Landé g | ΔE = g_J μ_B B m_j | `hamiltonians.landé_g`, `h_spin_orbit` | `docs/level_2_anomalous.tex` |
| 3 | Paschen-Back (strong field) | E ≈ μ_B B (m_l + g_s m_s) + ζ m_l m_s | `solvers.track_levels`, `animation.py` | `docs/level_3_paschen_back.tex` |
| 4 | Wigner-Eckart, selection rules, intensities, fitting | full line spectrum with polarisation | `transitions.py`, `fitting.py` | `docs/level_4_tensors.tex` |

All four levels are implemented and tested. Each has a notebook in `notebooks/` with saved output.

## The model

For one electron with orbital momentum $l$ and spin $1/2$:

$$H = H_0 + \zeta\,\mathbf{L}\cdot\mathbf{S} + \mu_B B\,(g_l L_z + g_s S_z)$$

It is built as a matrix in the basis $|l, m_l\rangle \otimes |\tfrac12, m_s\rangle$ and diagonalised
with `numpy.linalg.eigh`. The spin-orbit constant $\zeta$ is fixed once from the measured sodium
3p fine-structure splitting (17.196 cm⁻¹). Everything else is a prediction. Transition intensities
use the Wigner-Eckart theorem: the dipole matrix elements are 3-j symbols projected onto the exact
eigenvectors.

## Results (all reproduced by the test suite)

| Quantity | Model | Reference | Note |
|---|---|---|---|
| Na 3p fine-structure splitting | 17.196 cm⁻¹ | 17.196 cm⁻¹ | input, used to set ζ |
| g_J (²P₁/₂) | 0.6659 | 2/3 = 0.6667 | difference is g_s = 2.0023 instead of 2 |
| g_J (²P₃/₂) | 1.3341 | 4/3 = 1.3333 | same reason |
| Na D1 / D2 Zeeman components | 4 / 6 | 4 / 6 | textbook pattern |
| D1 shifts (units of μ_B B) | ±0.67 (π), ±1.33 (σ) | ±2/3, ±4/3 | weak field |
| D2 shifts (units of μ_B B) | ±0.33 (π), ±1.00, ±1.67 (σ) | ±1/3, ±1, ±5/3 | weak field |
| D2 : D1 intensity ratio | 2.000 | 2 | statistical weights 4 : 2 |
| ζ recovered from synthetic data (1 µeV noise) | about 0.01 % error | known input | `src/fitting.py`, Notebook 4 |

Reference values live in `data/experimental_reference.csv`. Please verify them against
[NIST ASD](https://physics.nist.gov/asd) before citing them.

## Quick start

```bash
git clone https://github.com/klinng/Quantum-Zeeman-Engine.git
cd Quantum-Zeeman-Engine
pip install -r requirements.txt
python -m pytest -q                       # 39 tests
python scripts/make_figures.py --gif      # figures and animation into docs/figures/
python scripts/execute_notebooks.py       # run the notebooks and save their output
cd docs && latexmk -pdf main.tex          # build the PDF report
```

Use it from Python:

```python
import numpy as np
from src.validation import zeta_from_fine_structure
from src.solvers import track_levels
from src.transitions import zeeman_lines
from src.constants import wavenumber_to_ev

zeta = zeta_from_fine_structure(17.196, l=1)               # eV, from sodium
levels = track_levels(1, np.linspace(0, 100, 200), zeta)   # shape (200, 6), eV

gap = wavenumber_to_ev(16956.17) + zeta                    # place 2P1/2 at the measured D1 energy
lines = zeeman_lines(1, 0, 1.0, zeta, 0.0, gap)            # p -> s at B = 1 T
print(lines[["polarisation", "wavenumber_cm1", "intensity"]])
```

## Project layout

```
src/            constants, hamiltonians, solvers, transitions, fitting, validation, animation, torch_engine
tests/          39 tests: algebra, Landé g, Paschen-Back, selection rules, sodium D lines, fitting
data/           experimental reference values
notebooks/      one lab per level, with explanations and saved output
docs/           LaTeX report (main.tex + one file per level), figures, and CODE_GUIDE.md
scripts/        make_figures.py, make_notebooks.py, execute_notebooks.py
.github/        CI: tests, notebook execution and PDF build on every push
```

New to the code? Read [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md): it explains every script and function.

## Roadmap

- [x] Levels 1 to 4: Zeeman physics, spectra, selection rules, fitting
- [ ] Add hydrogen n = 2 (2s and 2p levels together)
- [ ] Add a two-electron system (e.g. helium or calcium) with LS-coupling mixing
- [ ] Compare full spectra with published high-field measurements
- [ ] Test the optional PyTorch engine and use it for many-parameter fits

## License

MIT. See `LICENSE`.
