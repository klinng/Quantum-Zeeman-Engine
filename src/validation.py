"""Compare simulation output against known experimental values.

Reference data live in data/experimental_reference.csv. The key idea of the project:
if the numerical result reproduces the measured number, the model (our "root") is correct.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import ev_to_wavenumber, wavenumber_to_ev
from .hamiltonians import h_spin_orbit
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "experimental_reference.csv"


def load_reference() -> pd.DataFrame:
    return pd.read_csv(DATA, comment="#")


def zeta_from_fine_structure(delta_cm1: float, l: int = 1) -> float:
    """Spin-orbit constant zeta [eV] from the measured fine-structure splitting [cm^-1].

    For one electron, E(j=l+1/2) - E(j=l-1/2) = zeta * (l + 1/2).
    """
    return wavenumber_to_ev(delta_cm1) / (l + 0.5)


def fine_structure_splitting_cm1(l: int, zeta_ev: float) -> float:
    """Splitting between the two j levels [cm^-1] from the numerical H_SO eigenvalues."""
    w = np.unique(np.round(np.linalg.eigvalsh(h_spin_orbit(l, zeta_ev)), 14))
    return ev_to_wavenumber(w.max() - w.min())


def compare(name: str, computed: float, reference: float, unit: str = "") -> dict:
    err = computed - reference
    return {"quantity": name, "computed": computed, "reference": reference, "unit": unit,
            "abs_error": err, "rel_error_%": 100 * err / reference if reference else float("nan")}
