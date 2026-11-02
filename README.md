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
  <img src="docs/figures/p_orbital_levels.png" width="560" alt="Energy levels of a p electron versus magnetic field, from weak field to Paschen-Back">
</p>

*Energy levels of a p electron (sodium 3p spin-orbit constant) from the weak-field regime to
Paschen-Back. Levels with different m_j cross; levels with the same m_j repel.*

## Why decompose a problem?

1. **Understand** the phenomenon, the math and the physics behind it, one layer at a time.
2. **Verify** each layer against known results before adding the next, so errors can't hide.
3. **Compute, plot, animate** and run numerical experiments on a model you trust.

## The levels

| Level | Physics | Key result | Code | Report |
|---|---|---|---|---|
| 1 | Normal Zeeman (orbital only) | ΔE = μ_B B m_l | `hamiltonians.h_zeeman` | `docs/level_1_normal.tex` |
| 2 | Spin-orbit coupling, Landé g | ΔE = g_J μ_B B m_j | `hamiltonians.landé_g`, `h_spin_orbit` | `docs/level_2_anomalous.tex` |
| 3 | Paschen-Back (strong field) | E ≈ μ_B B (m_l + g_s m_s) + ζ m_l m_s | `solvers.track_levels`, `animation.py` | `docs/level_3_paschen_back.tex` |
| 4 | Wigner-Eckart, tensors, fitting ζ | selection rules, intensities | `torch_engine.py` (optional) | `docs/level_4_tensors.tex` |

Levels 1 to 3 are implemented and tested. **Level 4 is planned.**

## The model

For one electron with orbital momentum $l$ and spin $1/2$:

$$H = H_0 + \zeta\,\mathbf{L}\cdot\mathbf{S} + \mu_B B\,(g_l L_z + g_s S_z)$$

It is built as a matrix in the basis $|l, m_l\rangle \otimes |\tfrac12, m_s\rangle$ and diagonalised
with `numpy.linalg.eigh`. The spin-orbit constant $\zeta$ is fixed once from the measured sodium
3p fine-structure splitting (17.196 cm⁻¹). Everything else is a prediction.

## Results

| Quantity | Model | Reference | Note |
|---|---|---|---|
| Na 3p fine-structure splitting | 17.196 cm⁻¹ | 17.196 cm⁻¹ | input, used to set ζ |
| g_J (²P₁/₂) | 0.6659 | 2/3 = 0.6667 | difference is g_s = 2.0023 instead of 2 |
| g_J (²P₃/₂) | 1.3341 | 4/3 = 1.3333 | same reason |
| Stretched states at any B | exact to 1e-10 | analytic | `tests/test_physical_constants.py` |

Reference values live in `data/experimental_reference.csv`. Please verify them against
[NIST ASD](https://physics.nist.gov/asd) before citing them.

## Quick start

```bash
git clone https://github.com/klinng/Quantum-Zeeman-Engine.git
cd Quantum-Zeeman-Engine
pip install -r requirements.txt
python -m pytest -q                    # 26 tests
python scripts/make_figures.py --gif   # figures and animation into docs/figures/
cd docs && latexmk -pdf main.tex       # build the PDF report
```

Use it from Python:

```python
from src.validation import zeta_from_fine_structure
from src.solvers import track_levels
import numpy as np

zeta = zeta_from_fine_structure(17.196, l=1)      # eV, from sodium
b = np.linspace(0, 100, 200)                      # tesla
levels = track_levels(1, b, zeta)                 # shape (200, 6), eV
```

## Project layout

```
src/            physics code (constants, hamiltonians, solvers, validation, animation, torch_engine)
tests/          Hermiticity, angular-momentum algebra, Landé g, Paschen-Back limit, sodium check
data/           experimental reference values
notebooks/      one lab per level
docs/           LaTeX report (main.tex + one file per level) and generated figures
scripts/        make_figures.py, make_notebooks.py
.github/        CI: runs tests and builds the PDF on every push
```

## Roadmap

- [ ] Level 4: 3-j symbols, relative line intensities, selection rules
- [ ] Fit ζ from data with PyTorch autograd
- [ ] Add hydrogen n = 2 and a two-electron system
- [ ] Compare full spectra (all transitions) with published Zeeman measurements

## License

MIT. See `LICENSE`.
