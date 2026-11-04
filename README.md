# Quantum Zeeman Engine

[![Tests and PDF](https://github.com/klinng/Quantum-Zeeman-Engine/actions/workflows/latex.yml/badge.svg)](https://github.com/klinng/Quantum-Zeeman-Engine/actions/workflows/latex.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-99%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**Take one big physics problem, split it into levels, and make every level checkable:**
a LaTeX derivation, a Python model, a plot or animation, and a comparison with measured values.
If the computed number reproduces the experiment, the model underneath is correct.

The first case study is the **Zeeman effect** (how a magnetic field splits atomic energy levels).
It grew from four levels of theory into a small lab: sodium and hydrogen, real spectra,
a virtual spectrometer with colours, a prism and polarisation filters, and seven notebooks.

<p align="center">
  <img src="docs/figures/p_orbital_levels.png" width="500" alt="Energy levels of a p electron versus magnetic field, from weak field to Paschen-Back">
</p>

## Quick start

```bash
git clone https://github.com/klinng/Quantum-Zeeman-Engine.git
cd Quantum-Zeeman-Engine
pip install -r requirements.txt
python scripts/run_all.py --fast     # tests, figures, notebooks, PDF (drop --fast to also render the GIFs)
python -m pytest -q                  # 99 tests
```

Or only what you need: `python scripts/make_figures.py --gif`, `python scripts/make_optics_figures.py --gif`,
`python scripts/execute_notebooks.py`, `cd docs && latexmk -pdf main.tex`.

## What works for which atom

| System | Energy levels vs B | Line spectrum | Polarisation filters | Prism / instruments | Notebook | Report section |
|---|:-:|:-:|:-:|:-:|:-:|---|
| **Sodium** (3p to 3s, D lines) | yes | yes | yes | yes | 1 to 4, 6 | Levels 1 to 4, Extension B |
| **Hydrogen n = 2** (Lamb shift, 2s and 2p) | yes, with crossings | no (no jump between 2s and 2p) | no | no | 5 | Extension A |
| **Hydrogen H-alpha** (n = 3 to 2, 656 nm) | yes | yes | yes | not yet | 7 | Extension C |

A spectrum needs a jump between two shells. That is why hydrogen n = 2 alone has levels but no light, and why H-alpha was added.

## Gallery

<p align="center">
  <img src="docs/figures/na_d_lines_zeeman.png" width="760" alt="Computed Zeeman components of the sodium D1 and D2 lines with polarisation and intensity">
</p>

*Sodium D lines at 1 T: 4 components for D1, 6 for D2, with polarisation (colour) and relative intensity (height).
Everything comes from diagonalising the Hamiltonian and applying the dipole operator.*

<p align="center">
  <img src="docs/figures/halpha_regimes.png" width="820" alt="Hydrogen H-alpha at three field strengths through no filter, a parallel polariser and a perpendicular polariser">
</p>

*Hydrogen H-alpha at 0.05, 1 and 6 T. Rows: no filter, polariser parallel to B (keeps pi), polariser perpendicular to B (keeps sigma).
Grey is the 300 K Doppler width (8.1 pm). At 6 T the line has become the normal three-group Zeeman pattern.*

<p align="center">
  <img src="docs/figures/hydrogen_n2_levels.png" width="520" alt="Hydrogen n=2 levels versus magnetic field with the 2S-2P crossing marked">
</p>

*Hydrogen n = 2. The dotted line is the predicted 2S1/2 / 2P1/2 crossing at 573 G. A magnetic field never mixes 2s with 2p, so these levels cross instead of repelling.*

### The virtual spectrometer: colours, a prism, polarisation filters

<p align="center">
  <img src="docs/figures/optics_polarisation_filters.png" width="780" alt="Sodium D lines at 2 tesla through polarisers across the field and circular analysers along it">
</p>

*Sodium D lines at 2 T. Across the field a polariser parallel to B keeps only the pi lines and a perpendicular one keeps only sigma.
Along the field there is no pi light, and a circular analyser separates sigma+ (red) from sigma- (green).*

| Animation | What it shows |
|---|---|
| ![filter scan](docs/figures/optics_filter_scan.gif) | **Colour filter sweeping the spectrum**: only the lamp lines inside the pass band reach the prism screen |
| ![polariser](docs/figures/optics_polariser_rotation.gif) | **Rotating polariser** (Malus's law): pi and sigma lines trade places |
| ![view angle](docs/figures/optics_view_angle.gif) | **Tilting the line of sight**: pi fades out, sigma becomes circular |
| ![field ramp](docs/figures/optics_zeeman_field_ramp.gif) | **Field ramp**: the Zeeman components of Na D1 and D2 fan out |
| ![sodium to Paschen-Back](docs/figures/optics_sodium_paschen_back.gif) | **Sodium D lines up to 60 T**: two anomalous patterns collapse towards the normal Zeeman triplet (above B* = 24.6 T the field beats spin-orbit coupling) |
| ![H-alpha ramp](docs/figures/halpha_field_ramp.gif) | **Hydrogen H-alpha field ramp**: through no filter, a parallel and a perpendicular polariser, from weak field to Paschen-Back |

**An honest scale note.** The outermost sodium D2 component moves only about 27 pm per tesla, and every component is the same yellow.
A prism cannot separate them: the closest components need a resolving power of about 55 000 at 1 T, while a 5 cm glass prism offers about 2 000.

<p align="center">
  <img src="docs/figures/optics_instrument_ladder.png" width="780" alt="Resolving power of prisms, gratings and a Fabry-Perot against what the Zeeman pattern needs">
</p>

**The instrument that can.** A Fabry-Perot etalon (5 mm gap, finesse 30) turns each wavelength into a ring. At 0.3 T the sodium D2 ring splits
into six: two pi (blue) and four sigma (red and green), with peak heights close to the classic 4 : 3 : 1. A perpendicular polariser removes the pi pair.

<p align="center">
  <img src="docs/figures/optics_fabry_perot_rings.png" width="640" alt="Fabry-Perot ring images of sodium D2 at three fields, with and without a polariser">
</p>
<p align="center">
  <img src="docs/figures/optics_fabry_perot_profile.png" width="560" alt="Ring brightness against angle: six resolved Zeeman components coloured by polarisation">
</p>

**Strong field.** Above B* = ζ/μ_B ≈ 24.6 T the field overpowers the spin-orbit coupling and the sodium D doublet collapses towards the normal Zeeman
triplet. D1 and D2 lines cross over, so shifts are measured from the doublet's centre of gravity. A built-in check: the intensity-weighted mean σ shift is
±16.2 pm/T at every field from 1 to 120 T, whatever the pattern looks like.

<p align="center">
  <img src="docs/figures/optics_sodium_paschen_back_fan.png" width="620" alt="Sodium D line components against field from weak field to 60 tesla">
</p>

## Why decompose a problem?

1. **Understand** the phenomenon, the math and the physics behind it, one layer at a time.
2. **Verify** each layer against known results before adding the next, so errors can't hide.
3. **Compute, plot, animate** and run numerical experiments on a model you trust.

## The levels and extensions

| Part | Physics | Key result | Code | Report |
|---|---|---|---|---|
| Level 1 | Normal Zeeman (orbital only) | ΔE = μ_B B m_l, Lorentz triplet | `hamiltonians.h_zeeman` | `level_1_normal.tex` |
| Level 2 | Spin-orbit coupling, Landé g | ΔE = g_J μ_B B m_j | `hamiltonians.landé_g`, `h_spin_orbit` | `level_2_anomalous.tex` |
| Level 3 | Paschen-Back (strong field) | E ≈ μ_B B (m_l + g_s m_s) + ζ m_l m_s | `solvers.track_levels` | `level_3_paschen_back.tex` |
| Level 4 | Wigner-Eckart, selection rules, intensities, fitting | full line spectrum with polarisation | `transitions.py`, `fitting.py` | `level_4_tensors.tex` |
| Extension A | Hydrogen n = 2: Lamb shift, 2S-2P crossing | 573 G, against "about 570 G" in the literature | `hydrogen.py` | `ext_hydrogen_n2.tex` |
| Extension B | Colours, prism, polarisation filters, instrument ladder | four animations, required resolving power | `src/optics/` | `ext_optics.tex` |
| Extension C | Hydrogen H-alpha (n = 3 to 2) with radial integrals, Dirac fine structure | spectrum from weak field to Paschen-Back | `hydrogen_spectrum.py` | `ext_hydrogen_halpha.tex` |

Every part has a test file; notebooks 1 to 7 show the same ideas with explanations and saved output.

## The model

For one electron with orbital momentum $l$ and spin $1/2$:

$$H = H_0 + \zeta\,\mathbf{L}\cdot\mathbf{S} + \mu_B B\,(g_l L_z + g_s S_z)$$

It is built as a matrix in the basis $|l, m_l\rangle \otimes |\tfrac12, m_s\rangle$ and diagonalised with `numpy.linalg.eigh`.
For sodium, ζ is fixed once from the measured 3p fine-structure splitting (17.196 cm⁻¹); everything else is a prediction.
Transition intensities use the Wigner-Eckart theorem: dipole matrix elements are 3-j symbols projected onto the exact eigenvectors.
For hydrogen, each shell is block-diagonal in $l$, Dirac energies fix the fine structure, and the radial dipole integrals come from the wavefunctions.

## Results (all reproduced by the test suite)

### Sodium

| Quantity | Model | Reference | Note |
|---|---|---|---|
| 3p fine-structure splitting | 17.196 cm⁻¹ | 17.196 cm⁻¹ | input, used to set ζ |
| g_J (²P₁/₂) | 0.6659 | 2/3 = 0.6667 | difference is g_s = 2.0023 instead of 2 |
| g_J (²P₃/₂) | 1.3341 | 4/3 = 1.3333 | same reason |
| D1 / D2 Zeeman components | 4 / 6 | 4 / 6 | textbook pattern |
| D2 : D1 intensity ratio | 2.000 | 2 | statistical weights 4 : 2 |
| D1 components across B | 4 equal | 4 equal | classic result |
| D2 components across B (pi : inner sigma : outer sigma) | 4.00 : 2.99 : 1.00 | 4 : 3 : 1 | classic result |
| ζ recovered from synthetic data (1 µeV noise) | about 0.01 % error | known input | `fitting.py`, notebook 4 |
| Resolving power to separate components at 1 T | about 55 000 | prism 2 000, grating 1.2×10⁵ | a prism cannot show the pattern |

### Hydrogen

| Quantity | Model | Reference | Note |
|---|---|---|---|
| 2S₁/₂(−½) × 2P₁/₂(+½) crossing | 573.4 G | about 570 G | literature includes hyperfine structure |
| 2P₃/₂ − 2P₁/₂ (Dirac vs measured) | 10949 MHz | 10969 MHz | 0.2 % apart (QED); measured value not yet source-checked |
| ⟨2p\|r\|1s⟩ | 1.290266 a₀ | 128√6/243 a₀ | exact |
| A(2p to 1s), A(3p to 2s), A(3d to 2p) | 6.258e8, 2.242e7, 6.458e7 s⁻¹ | 6.265e8, 2.245e7, 6.465e7 s⁻¹ | about 0.1 %; reference values quoted from memory |
| H-alpha sigma shift at 20 T | 402.27 pm | 20.11 pm/T × B = 402.20 pm | normal Zeeman limit |
| H-alpha thermal Doppler width, 300 K | 8.1 pm | 8.1 pm | compare with 20.1 pm/T Zeeman shift |

Reference values live in `data/experimental_reference.csv`.

## Limits and honesty notes

- **Not modelled:** hyperfine structure, the diamagnetic term (about 0.02 cm⁻¹ at 20 T for n = 3), and electron-electron effects (one-electron model).
- **Approximation:** the n = 3 Lamb shift is the measured n = 2 value scaled as 1/n³.
- **Unchecked inputs:** the measured 2P fine structure (10969.04 MHz) and the decay-rate reference values were typed from memory. Check them against [NIST ASD](https://physics.nist.gov/asd) before citing.
- **Optics package:** `src/optics/` (colour, prism, polarisation, spectrometer, figures) was already in the repository when I joined this work and I don't know where it came from. Review it before relying on it. `instruments.py` and the H-alpha figures were written here.
- **Optional PyTorch engine** (`torch_engine.py`) has not been tested in this environment.

## Use it from Python

```python
import numpy as np
from src.validation import zeta_from_fine_structure
from src.solvers import track_levels
from src.hydrogen_spectrum import balmer_components
from src.optics.spectrometer import observe
from src.optics.polarization import linear_polariser

zeta = zeta_from_fine_structure(17.196, l=1)               # eV, from sodium
levels = track_levels(1, np.linspace(0, 100, 200), zeta)   # shape (200, 6), eV

halpha = balmer_components(1.0)                            # hydrogen H-alpha at 1 T
seen = observe(halpha, 90.0, linear_polariser(0.0))        # across the field, polariser parallel to B
print(seen[seen.observed > 1e-3][["polarisation", "shift_pm", "observed"]])
```

## Project layout

```
src/            constants, hamiltonians, solvers, transitions, fitting, hydrogen, hydrogen_spectrum, validation, animation, torch_engine
src/optics/     color, prism, polarization, spectrometer, instruments, fabry_perot, sodium_paschen_back, figures, hydrogen_figures
tests/          99 tests: algebra, Landé g, Paschen-Back, selection rules, sodium D lines, fitting, hydrogen n=2, H-alpha, Fabry-Perot, optics
data/           experimental reference values
notebooks/      seven labs (levels 1 to 4, hydrogen n=2, virtual spectrometer, H-alpha), with explanations and saved output
docs/           LaTeX report (main.tex + one file per part), figures, and CODE_GUIDE.md
scripts/        run_all.py, make_figures.py, make_optics_figures.py, make_notebooks.py, execute_notebooks.py
.github/        CI: tests, notebook execution and PDF build on every push
```

New to the code? Read [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md): it explains every script and function.

## Roadmap

- [x] Levels 1 to 4: Zeeman physics, spectra, selection rules, fitting
- [x] Hydrogen n = 2 (2s and 2p together, Lamb shift)
- [x] Virtual spectrometer: colours, prism, polarisation filters, instrument ladder
- [x] Hydrogen H-alpha: a real hydrogen spectrum with polarisation filters
- [x] Animate the H-alpha field ramp through the polarisation filters
- [ ] Push H-alpha through the prism and the instrument ladder
- [x] Show the Zeeman pattern in a simulated Fabry-Perot ring image (the instrument that can actually resolve it)
- [x] Paschen-Back collapse of the sodium D lines (60 T) in the optics animations
- [ ] Electric field mixing 2s with 2p (Lamb-Retherford quenching, anticrossings)
- [ ] A two-electron system (helium or calcium) with LS-coupling mixing
- [ ] Compare full spectra with published high-field measurements
- [ ] Test the optional PyTorch engine and use it for many-parameter fits

## Sources

- Zeeman and Paschen-Back theory: standard quantum mechanics texts and the UCSD PH130A notes.
- Hydrogen level crossings and the Lamb shift: arXiv:1104.4421.
- Glass dispersion: Sellmeier coefficients from the Schott catalogue.

## License

MIT. See `LICENSE`.
