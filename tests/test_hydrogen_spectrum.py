"""Tests for src/hydrogen_spectrum.py (H-alpha in a magnetic field)."""
import numpy as np
import pytest
from scipy.integrate import quad

from src.hydrogen_spectrum import (ALPHA, RY_H_CM1, MHZ_TO_CM1, balmer_components, doppler_fwhm_pm, dirac_energy_cm1,
                                   einstein_a, manifold_hamiltonian, radial_function, radial_integral)
from src.optics.polarization import linear_polariser
from src.optics.spectrometer import observe

MU_B_SHIFT_PM_PER_T = 20.11   # normal Zeeman shift of 656.28 nm: lambda^2 * mu_B / (h c) per tesla


@pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_radial_function_is_normalised(n, l):
    val = quad(lambda r: float(radial_function(n, l, r)) ** 2 * r ** 2, 0, 80 * n * n, limit=400)[0]
    assert val == pytest.approx(1.0, abs=1e-6)


def test_lyman_alpha_radial_integral_is_exact():
    assert radial_integral(2, 1, 1, 0) == pytest.approx(128 * np.sqrt(6) / 243, rel=1e-8)


def test_einstein_a_values_match_literature():
    # reference values: Wiese and Fuhr, J. Phys. Chem. Ref. Data 38, 565 (2009), Table 6
    assert einstein_a(2, 1, 1, 0) == pytest.approx(6.265e8, rel=0.01)
    assert einstein_a(3, 0, 2, 1) == pytest.approx(6.317e6, rel=0.01)
    assert einstein_a(3, 1, 2, 0) == pytest.approx(2.245e7, rel=0.01)
    assert einstein_a(3, 2, 2, 1) == pytest.approx(6.465e7, rel=0.01)
    assert einstein_a(3, 0, 2, 0) == 0.0          # no s -> s dipole transition


def test_fine_structure_of_n2():
    split = (dirac_energy_cm1(2, 1.5) - dirac_energy_cm1(2, 0.5))
    assert split == pytest.approx(ALPHA ** 2 * RY_H_CM1 / 16, rel=1e-9)


def test_zero_field_level_count():
    count = lambda n, lamb: len(set(np.round(np.linalg.eigvalsh(manifold_hamiltonian(n, 0.0, lamb)), 12)))
    assert count(3, True) == 4 and count(3, False) == 3
    assert count(2, True) == 3 and count(2, False) == 2


def test_lamb_shift_n2():
    from src.constants import ev_to_wavenumber
    lamb = manifold_hamiltonian(2, 0.0, True)[0, 0].real - manifold_hamiltonian(2, 0.0, False)[0, 0].real
    assert ev_to_wavenumber(lamb) / MHZ_TO_CM1 == pytest.approx(1057.845, rel=1e-6)


def test_total_rate_is_equal_for_the_three_polarisations():
    df = balmer_components(2.0, threshold=1e-14)
    totals = df.groupby("q")["rate"].sum()
    assert totals.max() == pytest.approx(totals.min(), rel=1e-6)


def test_strong_field_sigma_shift_is_normal_zeeman():
    b = 20.0
    df = balmer_components(b, threshold=1e-9)
    mean = lambda q: np.average(df[df.q == q].shift_pm, weights=df[df.q == q].rate)
    half = (mean(-1) - mean(1)) / 2
    assert half == pytest.approx(MU_B_SHIFT_PM_PER_T * b, rel=0.03)


def test_polariser_along_field_keeps_only_pi():
    df = observe(balmer_components(1.0), 90.0, linear_polariser(0.0))
    kept = df[df["observed"] > 1e-9 * df["observed"].max()]
    assert set(kept.q) == {0}


def test_zero_or_negative_field_rejected():
    with pytest.raises(ValueError):
        balmer_components(0.0)
    with pytest.raises(ValueError):
        balmer_components(-1.0)


def test_doppler_width():
    assert doppler_fwhm_pm() == pytest.approx(8.1, abs=0.1)
