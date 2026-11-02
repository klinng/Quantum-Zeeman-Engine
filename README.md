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

<p align="center">
  <img src="docs/figures/hydrogen_n2_levels.png" width="520" alt="Hydrogen n=2 levels versus magnetic field with the 2S-2P crossing marked">
</p>

*Hydrogen n = 2 (2S in red, 2P₁/₂ in blue, 2P₃/₂ in green). The dotted line is the predicted
2S₁/₂ / 2P₁/₂ crossing at 573 G. A magnetic field never mixes 2s with 2p, so these levels cross
instead of repelling.*

## The virtual spectrometer: colours, a prism, and polarisation filters

Put optics in front of the computed lines and watch what each filter lets through (Extension B, `src/optics/`).

<p align="center">
  <img src="docs/figures/optics_polarisation_filters.png" width="780" alt="Sodium D lines at 2 tesla through polarisers across the field and circular analysers along it">
</p>

*Sodium D lines at 2 T. Top row, looking across the field: a polariser parallel to B keeps only the π lines,
one perpendicular keeps only the σ lines. Bottom row, looking along the field: no π light, and a circular analyser
separates σ+ (red) from σ− (green). The yellow strip is the colour of the light itself.*

| Animation | What it shows |
|---|---|
| ![filter scan](docs/figures/optics_filter_scan.gif) | **Colour filter sweeping the spectrum**: only the lamp lines inside the pass band reach the prism screen |
| ![polariser](docs/figures/optics_polariser_rotation.gif) | **Rotating polariser** (Malus's law): π and σ lines trade places |
| ![view angle](docs/figures/optics_view_angle.gif) | **Tilting the line of sight**: π fades out, σ becomes circular |
| ![field ramp](docs/figures/optics_zeeman_field_ramp.gif) | **Field ramp**: the Zeeman components of Na D1 and D2 fan out |

**An honest scale note.** The Zeeman shift is tiny: about 27 pm per tesla for the outermost sodium D2 line, and every
component is the same yellow. A prism cannot separate them. The closest components need a resolving power of about
55 000 at 1 T, while a 5 cm glass prism offers about 2 000:

<p align="center">
  <img src="docs/figures/optics_instrument_ladder.png" width="780" alt="Resolving power of prisms, gratings and a Fabry-Perot against what the Zeeman pattern needs">
</p>

Reproduced from the model with nothing typed in: the four D1 components are equally bright and the D2 components
have relative intensities 4 : 3 : 1 when viewed across the field.

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

**Extension A: hydrogen n = 2** (`hydrogen.py`, notebook 5, `docs/ext_hydrogen_n2.tex`) puts the 2s and
2p levels in one Hamiltonian, adds the Lamb shift, and predicts the 2S-2P level-crossing field
(573 G, against "about 570 G" in the literature).

**Extension B: the virtual spectrometer** (`src/optics/`, notebook 6, `docs/ext_optics.tex`) adds colours, a glass prism,
polarisation filters and an instrument ladder, with four animations.

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
| H 2S₁/₂(−½) × 2P₁/₂(+½) crossing | 573.4 G | about 570 G | literature includes hyperfine structure; Lamb shift is the only input |
| H 2P₃/₂ − 2P₁/₂ from Dirac theory vs measured | 10949 MHz | 10969 MHz | 0.2 % apart (QED); the measured value is not yet source-checked |
| Na D1 components across B (weak field) | 4 equal | 4 equal | classic result, from matrix elements and the angular pattern |
| Na D2 components across B (π : inner σ : outer σ) | 4.00 : 2.99 : 1.00 | 4 : 3 : 1 | classic result |
| Resolving power to separate Zeeman components at 1 T | about 55 000 | prism 2 000, grating 1.2×10⁵ | a prism cannot show the pattern |

Reference values live in `data/experimental_reference.csv`. Please verify them against
[NIST ASD](https://physics.nist.gov/asd) before citing them.

## Quick start

```bash
git clone https://github.com/klinng/Quantum-Zeeman-Engine.git
cd Quantum-Zeeman-Engine
pip install -r requirements.txt
python scripts/run_all.py                 # or do everything in one go (add --fast to skip GIFs)
python -m pytest -q                       # 68 tests
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
src/            constants, hamiltonians, solvers, transitions, fitting, hydrogen, validation, animation, torch_engine
src/optics/     color, prism, polarization, spectrometer, instruments, figures (the virtual spectrometer)
tests/          68 tests: algebra, Landé g, Paschen-Back, selection rules, sodium D lines, fitting, hydrogen n=2, optics
data/           experimental reference values
notebooks/      six labs (levels 1 to 4, hydrogen, virtual spectrometer), with explanations and saved output
docs/           LaTeX report (main.tex + one file per level), figures, and CODE_GUIDE.md
scripts/        run_all.py, make_figures.py, make_optics_figures.py, make_notebooks.py, execute_notebooks.py
.github/        CI: tests, notebook execution and PDF build on every push
```

New to the code? Read [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md): it explains every script and function.

## Roadmap

- [x] Levels 1 to 4: Zeeman physics, spectra, selection rules, fitting
- [x] Add hydrogen n = 2 (2s and 2p levels together, Lamb shift)
- [x] Virtual spectrometer: colours, prism, polarisation filters, instrument ladder
- [ ] Show the Zeeman pattern in a simulated Fabry-Perot ring image (the instrument that can actually resolve it)
- [ ] Add the Paschen-Back collapse of the D lines (60 T) to the optics animations
- [ ] Add an electric field to mix 2s with 2p (Lamb-Retherford quenching, anticrossings)
- [ ] Add a two-electron system (e.g. helium or calcium) with LS-coupling mixing
- [ ] Compare full spectra with published high-field measurements
- [ ] Test the optional PyTorch engine and use it for many-parameter fits

## License

MIT. See `LICENSE`.
