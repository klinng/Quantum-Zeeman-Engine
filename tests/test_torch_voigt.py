"""Tests for src/torch_voigt.py. The NumPy tests always run; the PyTorch tests are skipped when torch is not installed."""
import numpy as np
import pytest
from scipy.special import voigt_profile, wofz

from src.constants import wavenumber_to_ev
from src.hyperfine_lines import FINE_NA_CM1
from src.torch_voigt import (doppler_sigma_hz, faddeeva, line_list, lorentz_hwhm_hz, voigt, zeeman_spectrum)
from src.transitions import zeeman_lines

ZETA = wavenumber_to_ev(FINE_NA_CM1) * 2 / 3
ARGS = dict(zeta_ev=ZETA, nu0_hz=5.0885e14, mass_amu=22.98977, a_total_per_s=0.616e8, k_pressure_hz_per_pa=1.0e4)


def test_faddeeva_matches_scipy_upper_half_plane():
    rng = np.random.default_rng(1)
    z = rng.uniform(-30, 30, 4000) + 1j * rng.uniform(0, 30, 4000)
    assert np.max(np.abs(faddeeva(z) - wofz(z))) < 1e-11


@pytest.mark.parametrize("sigma,gamma", [(0.01, 0.0), (0.3, 0.01), (1.0, 0.5), (5.0, 3.0), (1.0, 30.0)])
def test_voigt_matches_scipy(sigma, gamma):
    x = np.linspace(-60, 60, 3001)
    ref = voigt_profile(x, sigma, gamma)
    assert np.max(np.abs(voigt(x, sigma, gamma) - ref)) < 1e-12 * max(1.0, ref.max())


def test_voigt_has_unit_area():
    x = np.linspace(-4000, 4000, 800001)
    assert np.trapezoid(voigt(x, 1.0, 0.7), x) == pytest.approx(1.0, abs=2e-3)       # Lorentzian tails beyond +-4000 lost


def test_doppler_width_scales_as_sqrt_t():
    assert doppler_sigma_hz(5e14, 400, 23) / doppler_sigma_hz(5e14, 100, 23) == pytest.approx(2.0)


def test_lorentz_width_adds_pressure():
    assert lorentz_hwhm_hz(0.616e8, 100.0, 1e4) == pytest.approx(0.616e8 / (4 * np.pi) + 1e6)


def test_line_list_matches_existing_solver():
    f, s = line_list(0.5, ZETA)
    df = zeeman_lines(1, 0, 0.5, ZETA)
    keep = s > 1e-10
    assert np.allclose(np.sort(f[keep]), np.sort(df.energy_ev.values / (6.62607015e-34 / 1.602176634e-19)))
    assert np.allclose(np.sort(s[keep]), np.sort(df.intensity.values))


def test_spectrum_is_normalised_to_number_of_upper_states():
    nu = np.linspace(-600e9, 600e9, 240001)
    s = zeeman_spectrum(nu, 0.5, 400.0, 0.0, **ARGS)
    assert np.trapezoid(s, nu) == pytest.approx(6.0, rel=2e-3)


def test_torch_voigt_agrees_with_numpy_and_gradients_are_correct():
    torch = pytest.importorskip("torch")
    x = torch.linspace(-10, 10, 41, dtype=torch.float64)
    sig = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    gam = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    v = voigt(x, sig, gam)
    assert np.allclose(v.detach().numpy(), voigt(x.numpy(), 1.3, 0.4), atol=1e-14)
    assert torch.autograd.gradcheck(lambda s, g: voigt(x, s, g), (sig, gam), eps=1e-6, atol=1e-6)


def test_torch_gradients_wrt_b_t_p_match_finite_differences():
    torch = pytest.importorskip("torch")
    # Frequencies on the core of a line: far in the wings the temperature dependence is a tiny signal on a large background
    # and the finite difference is dominated by rounding noise (it changed in the 4th digit with the step).
    centre = float(np.sort(line_list(0.5, ZETA)[0])[4])
    nu = torch.as_tensor(centre + np.linspace(-1.5e9, 1.5e9, 7), dtype=torch.float64)
    base = dict(B=0.5, T=400.0, P=50.0)

    def f(b, t, p, backend="torch"):
        return zeeman_spectrum(nu if backend == "torch" else nu.numpy(), b, t, p, backend=backend, **ARGS)

    b, t, p = (torch.tensor(base[k], dtype=torch.float64, requires_grad=True) for k in ("B", "T", "P"))
    out = f(b, t, p)
    assert np.allclose(out.detach().numpy(), f(0.5, 400.0, 50.0, "numpy"), rtol=1e-8)
    for name, var, step in (("B", b, 1e-5), ("T", t, 1e-1), ("P", p, 1.0)):
        g = torch.autograd.grad(out.sum(), var, retain_graph=True)[0].item()
        hi = dict(base); hi[name] += step
        lo = dict(base); lo[name] -= step
        fd = (f(hi["B"], hi["T"], hi["P"], "numpy").sum() - f(lo["B"], lo["T"], lo["P"], "numpy").sum()) / (2 * step)
        assert g == pytest.approx(fd, rel=1e-4, abs=1e-30)
