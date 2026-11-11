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
    assert einstein_a(2, 1, 1, 0) == pytest.approx(6.2649e8, rel=2e-4)
    assert einstein_a(3, 0, 2, 1) == pytest.approx(6.3143e6, rel=2e-4)
    assert einstein_a(3, 1, 2, 0) == pytest.approx(2.2448e7, rel=2e-4)
    assert einstein_a(3, 2, 2, 1) == pytest.approx(6.4651e7, rel=2e-4)
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
    totals = df.groupby("q")["line_strength"].sum()      # pure wave-function quantity: exactly isotropic
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


# --- line strength versus physical emission rate (audit priority 2) -----------------------------

def test_component_rates_sum_to_manifold_einstein_a():
    """Sum of the per-component A over all components, divided by the 18 upper states, equals the
    degeneracy-weighted mean of the three l-resolved Einstein coefficients (agreement to the omega^3
    spread of the fine-structure components, ~1e-4)."""
    from src.hydrogen_spectrum import einstein_a
    df = balmer_components(0.5, threshold=1e-14)
    expected = (2 * einstein_a(3, 0, 2, 1) + 6 * einstein_a(3, 1, 2, 0) + 10 * einstein_a(3, 2, 2, 1)) / 18
    assert df.einstein_a.sum() / 18 == pytest.approx(expected, rel=2e-4)


def test_einstein_a_of_a_component_scales_as_omega_cubed_times_strength():
    """A_ik = (4/3) alpha^3 w^3 S_ik: the ratio A / (S w^3) must be one constant for all components."""
    from src.hydrogen_spectrum import RY_INF_CM1, RATE_AU_PER_S
    from src.optics.spectrometer import N_AIR
    df = balmer_components(2.0, threshold=1e-9)
    # balmer_components reports AIR wavelengths, lambda_air = 1e7 / (sigma_vac * N_AIR) with the constant index N_AIR
    # (optics.spectrometer.vacuum_cm1_to_air_nm). Invert exactly that convention to get the VACUUM wavenumber
    # in cm^-1, which is what the Hartree conversion w = sigma / (2 Ry_inf) needs. No tolerance hides a convention gap.
    sigma = 1e7 / (df.wavelength_nm * N_AIR)
    k = df.einstein_a / (df.line_strength * sigma ** 3)
    assert k.max() / k.min() - 1 < 1e-9                  # same constant for every component (rounding only)
    const = 4 / 3 * ALPHA ** 3 * RATE_AU_PER_S / (2 * RY_INF_CM1) ** 3
    assert k.mean() == pytest.approx(const, rel=1e-9)


def test_relative_rate_differs_from_line_strength_only_by_omega_cubed():
    df = balmer_components(1.0, threshold=1e-9)
    ratio = df["rate"] / df.line_strength
    assert abs(ratio.mean() - 1) < 1e-3 and ratio.std() < 2e-4


def test_einstein_a_against_wiese_fuhr_finite_mass_table():
    """Wiese and Fuhr, J. Phys. Chem. Ref. Data 38, 565 (2009), Table 6 (H I; finite-nuclear-mass convention,
    multiplet values): 2p-1s 6.2649e8, 3s-2p 6.3143e6, 2s-3p 2.2448e7, 2p-3d 6.4651e7 (1/s). The small residuals
    (<= 1.1e-4) are the relativistic/fine-structure terms of the tabulated values, absent from the
    nonrelativistic formula used here."""
    assert einstein_a(2, 1, 1, 0) == pytest.approx(6.2649e8, rel=2e-4)
    assert einstein_a(3, 0, 2, 1) == pytest.approx(6.3143e6, rel=2e-4)
    assert einstein_a(3, 1, 2, 0) == pytest.approx(2.2448e7, rel=2e-4)
    assert einstein_a(3, 2, 2, 1) == pytest.approx(6.4651e7, rel=2e-4)


# --- reduced-mass consistency (audit priority 4) -------------------------------------------------

def test_lyman_alpha_closed_form_and_reduced_mass_scaling():
    """Independent closed form (no radial integral, no quadrature): A(2p->1s, Z=1, infinite mass) =
    (2/3)^8 alpha^5 m_e c^2 / hbar. Finite nuclear mass multiplies it by mu/m_e = 1/(1 + m_e/m_p):
    transition energy ~ mu (cubed), dipole length ~ 1/mu (squared)."""
    import scipy.constants as sc
    from src.hydrogen_spectrum import A_MU_OVER_A0
    a_inf = (2 / 3) ** 8 * sc.alpha ** 5 * sc.m_e * sc.c ** 2 / sc.hbar
    assert a_inf == pytest.approx(6.26831e8, rel=1e-5)
    assert einstein_a(2, 1, 1, 0) == pytest.approx(a_inf / A_MU_OVER_A0, rel=1e-8)
    assert einstein_a(2, 1, 1, 0) / a_inf == pytest.approx(1 / (1 + sc.m_e / sc.m_p), rel=1e-8)


def test_reduced_mass_factor_is_the_same_for_every_transition():
    """A(finite)/A(infinite) = mu/m_e for any hydrogenic transition, so A ratios between lines are unchanged."""
    from src.hydrogen_spectrum import A_MU_OVER_A0, RATE_AU_PER_S
    for t in ((2, 1, 1, 0), (3, 0, 2, 1), (3, 1, 2, 0), (3, 2, 2, 1)):
        n_up, l_up, n_low, l_low = t
        w_inf = (1 / n_low ** 2 - 1 / n_up ** 2) / 2                     # hartree, infinite mass
        a_inf = 4 / 3 * ALPHA ** 3 * w_inf ** 3 * max(l_up, l_low) / (2 * l_up + 1) \
            * radial_integral(*t) ** 2 * RATE_AU_PER_S
        assert einstein_a(*t) / a_inf == pytest.approx(1 / A_MU_OVER_A0, rel=1e-8)


def test_radial_integral_is_in_units_of_the_reduced_mass_bohr_radius():
    """radial_integral keeps the exact hydrogenic numbers (128 sqrt6 / 243 for 2p-1s); the physical length in
    Bohr radii a_0 is that times a_mu/a_0 = 1 + m_e/m_p, which full_dipole applies."""
    from src.hydrogen_spectrum import A_MU_OVER_A0, full_dipole
    assert radial_integral(2, 1, 1, 0) == pytest.approx(128 * 6 ** 0.5 / 243, rel=1e-9)
    d = full_dipole(2, 1)
    assert np.abs(d[0]).max() == pytest.approx(radial_integral(2, 1, 1, 0) * A_MU_OVER_A0 / np.sqrt(3), rel=1e-9)
