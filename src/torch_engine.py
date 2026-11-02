"""Optional PyTorch engine: fit physical parameters (zeta) to measured data by autograd.

PyTorch is only needed for this module; the rest of the project runs on NumPy/SciPy.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch_engine needs PyTorch: pip install torch") from exc

from .constants import G_L, G_S, MU_B_EV
from .hamiltonians import coupled_ops


def _ops_torch(l: int):
    o = coupled_ops(l)
    return {k: torch.tensor(v, dtype=torch.complex128) for k, v in o.items()}


def energies(l: int, b_tesla: float, zeta: torch.Tensor) -> torch.Tensor:
    """Differentiable eigenvalues of H(B; zeta)."""
    o = _ops_torch(l)
    ls = o["Lx"] @ o["Sx"] + o["Ly"] @ o["Sy"] + o["Lz"] @ o["Sz"]
    h = zeta * ls + MU_B_EV * b_tesla * (G_L * o["Lz"] + G_S * o["Sz"])
    return torch.linalg.eigvalsh(h)


def fit_zeta(l: int, b_values, target_levels, zeta0: float = 1e-3, steps: int = 500, lr: float = 5e-2):
    """Fit zeta so that sorted eigenvalues match target_levels (shape (nB, 2(2l+1)), eV)."""
    scale = zeta0
    p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)   # zeta = p * scale
    target = torch.tensor(np.asarray(target_levels), dtype=torch.float64)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = torch.stack([energies(l, float(b), p * scale) for b in b_values])
        loss = ((pred - target) ** 2).mean() / scale**2
        loss.backward()
        opt.step()
    return float(p.detach() * scale)
