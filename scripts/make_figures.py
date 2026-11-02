"""Generate all figures used in docs/ and optionally the animation. Run from repo root:
    python scripts/make_figures.py [--gif]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.animation import animate_splitting, b_scale_tesla, plot_energy_levels  # noqa: E402
from src.validation import compare, fine_structure_splitting_cm1, zeta_from_fine_structure  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ZETA = zeta_from_fine_structure(17.196, l=1)   # sodium 3p
print(compare("Na 3p splitting", fine_structure_splitting_cm1(1, ZETA), 17.196, "cm^-1"))

b_max = 5 * b_scale_tesla(ZETA)
ax = plot_energy_levels(1, b_max, ZETA)
ax.figure.tight_layout()
ax.figure.savefig(FIG / "p_orbital_levels.pdf")
ax.figure.savefig(FIG / "p_orbital_levels.png", dpi=150)
plt.close(ax.figure)
print("saved figures to", FIG)

if "--gif" in sys.argv:
    animate_splitting(1, b_max, ZETA, save_path=str(FIG / "p_orbital_split.gif"))
    print("saved animation")
