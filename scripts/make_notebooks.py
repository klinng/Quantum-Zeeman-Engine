"""Create the four lab notebooks (run once; notebooks are then edited normally)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(exist_ok=True)

HEAD = "import sys; sys.path.insert(0, '..')\nimport numpy as np, matplotlib.pyplot as plt\n"


def nb(cells):
    def cell(kind, src):
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        return c
    return {"cells": [cell(k, s) for k, s in cells], "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}


NOTEBOOKS = {
    "01_normal_zeeman_lab.ipynb": [
        ("markdown", "# Level 1: Normal Zeeman effect\nWith spin-orbit off (zeta = 0) levels are equally spaced, dE = mu_B B m_l."),
        ("code", HEAD + "from src.solvers import energy_vs_field\nfrom src.constants import MU_B_EV\n"
                 "b = np.linspace(0, 10, 50)\ne = energy_vs_field(1, b, 0.0, g_s=0.0)  # g_s=0 -> orbital only\n"
                 "plt.plot(b, e*1e3); plt.xlabel('B (T)'); plt.ylabel('E (meV)'); plt.show()\n"
                 "print('spacing per tesla (eV):', (e[-1,-1]-e[-1,0])/2/10, ' mu_B:', MU_B_EV)"),
    ],
    "02_anomalous_zeeman_lab.ipynb": [
        ("markdown", "# Level 2: Anomalous Zeeman effect\nSet zeta from sodium and check Lande g-factors."),
        ("code", HEAD + "from src.validation import zeta_from_fine_structure\nfrom src.hamiltonians import landé_g\n"
                 "zeta = zeta_from_fine_structure(17.196, l=1)\nprint('zeta (eV):', zeta)\n"
                 "print('g(2P1/2), g(2P3/2):', landé_g(1, 0.5), landé_g(1, 1.5))"),
    ],
    "03_paschen_back_transition_lab.ipynb": [
        ("markdown", "# Level 3: Weak field to Paschen-Back"),
        ("code", HEAD + "from src.animation import plot_energy_levels, b_scale_tesla\nfrom src.validation import zeta_from_fine_structure\n"
                 "zeta = zeta_from_fine_structure(17.196, l=1)\nplot_energy_levels(1, 5*b_scale_tesla(zeta), zeta, show=True)"),
    ],
    "04_tensor_operators_torch.ipynb": [
        ("markdown", "# Level 4: Tensor operators and PyTorch fitting (planned)\nRequires `pip install torch`."),
        ("code", HEAD + "from src.solvers import energy_vs_field\nfrom src.torch_engine import fit_zeta\n"
                 "true_zeta = 1.42e-3\nb = np.linspace(0, 3, 8)\ndata = energy_vs_field(1, b, true_zeta)\n"
                 "print('fitted zeta:', fit_zeta(1, b, data, zeta0=1e-3), 'true:', true_zeta)"),
    ],
}

for name, cells in NOTEBOOKS.items():
    (OUT / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", name)
