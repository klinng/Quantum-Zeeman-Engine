"""Physical constants (SI / eV). Values come from scipy.constants (CODATA)."""
from scipy import constants as _c

HBAR = _c.hbar                                   # J s
MU_B_J = _c.physical_constants["Bohr magneton"][0]            # J / T
MU_B_EV = _c.physical_constants["Bohr magneton in eV/T"][0]   # eV / T
G_S = abs(_c.physical_constants["electron g factor"][0])      # 2.0023193...
G_L = 1.0                                        # orbital g-factor (infinite nuclear mass)
HC_EV_CM = _c.h * _c.c / _c.e * 100.0            # eV * cm  (so E[eV] = HC_EV_CM * wavenumber[cm^-1])


def wavenumber_to_ev(sigma_cm1: float) -> float:
    """Convert a wavenumber in cm^-1 to energy in eV."""
    return HC_EV_CM * sigma_cm1


def ev_to_wavenumber(e_ev: float) -> float:
    """Convert an energy in eV to a wavenumber in cm^-1."""
    return e_ev / HC_EV_CM
