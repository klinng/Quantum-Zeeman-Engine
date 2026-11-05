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

from src.animation import animate_splitting, b_scale_tesla, plot_energy_levels, plot_stick_spectrum  # noqa: E402
from src.constants import ev_to_wavenumber, wavenumber_to_ev  # noqa: E402
from src.transitions import zeeman_lines  # noqa: E402
from src.validation import compare, fine_structure_splitting_cm1, zeta_from_fine_structure  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ZETA = zeta_from_fine_structure(17.196, l=1)   # sodium 3p
print(compare("Na 3p splitting", fine_structure_splitting_cm1(1, ZETA), 17.196, "cm^-1"))

# --- Level 3: energy levels versus B -------------------------------------------------------
b_max = 5 * b_scale_tesla(ZETA)
ax = plot_energy_levels(1, b_max, ZETA)
ax.figure.tight_layout()
ax.figure.savefig(FIG / "p_orbital_levels.pdf")
ax.figure.savefig(FIG / "p_orbital_levels.png", dpi=150)
plt.close(ax.figure)

# --- Level 4: sodium D1 / D2 Zeeman patterns at B = 1 T ------------------------------------
D1_CM1 = 16956.17                                  # measured Na D1 wavenumber (see data/ csv)
gap = wavenumber_to_ev(D1_CM1) + ZETA              # 2P1/2 lies at E0 - zeta above 2S1/2
d1_centre, d2_centre = gap - ZETA, gap + ZETA / 2
df = zeeman_lines(1, 0, 1.0, ZETA, 0.0, gap)
mid = 0.5 * (d1_centre + d2_centre)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
plot_stick_spectrum(df[df.energy_ev < mid], ev_to_wavenumber(d1_centre), axes[0], "Na D1 (2P1/2 to 2S1/2), B = 1 T")
plot_stick_spectrum(df[df.energy_ev >= mid], ev_to_wavenumber(d2_centre), axes[1], "Na D2 (2P3/2 to 2S1/2), B = 1 T")
fig.tight_layout()
fig.savefig(FIG / "na_d_lines_zeeman.pdf")
fig.savefig(FIG / "na_d_lines_zeeman.png", dpi=150)
plt.close(fig)
# --- Extension A: hydrogen n = 2 ------------------------------------------------------------
import numpy as np  # noqa: E402

from src.animation import plot_n2_levels  # noqa: E402
from src.hydrogen import level_index, track_n2  # noqa: E402
from src.solvers import find_crossings  # noqa: E402

b_h, e_h, lab_h, _ = track_n2(np.linspace(1e-6, 0.5, 2500))
i_s, i_p = level_index(lab_h, "2S1/2", -0.5), level_index(lab_h, "2P1/2", 0.5)
b_cross = find_crossings(b_h, e_h[:, i_s], e_h[:, i_p])[0]
print(f"hydrogen 2S1/2(-1/2) x 2P1/2(+1/2) crossing: {b_cross * 1e4:.1f} G (literature: about 570 G)")
ax = plot_n2_levels(b_h, e_h, lab_h, mark_fields_gauss=[b_cross * 1e4])
ax.figure.tight_layout()
ax.figure.savefig(FIG / "hydrogen_n2_levels.pdf")
ax.figure.savefig(FIG / "hydrogen_n2_levels.png", dpi=150)
plt.close(ax.figure)
from src.literature import fig_literature_comparison  # noqa: E402
from src.two_electron import fit_to_levels  # noqa: E402
from src.two_electron_figures import fig_two_electron_levels  # noqa: E402
from src.stark_figures import fig_anticrossing, fig_stark_fan  # noqa: E402

for name, fn in (("hydrogen_stark_fan", fig_stark_fan), ("hydrogen_anticrossing", fig_anticrossing),
                 ("literature_sodium_groups", fig_literature_comparison)):
    f = fn()
    f.savefig(FIG / f"{name}.pdf")
    f.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(f)
_z, _d, _et, _ = fit_to_levels(15157.91, 15210.06, 15315.94, 23652.04)      # calcium 4s4p, energies from memory (unverified)
_f = fig_two_electron_levels(_z, _d, _et)
_f.savefig(FIG / "two_electron_calcium_levels.pdf")
_f.savefig(FIG / "two_electron_calcium_levels.png", dpi=150)
plt.close(_f)
print("saved figures to", FIG)

if "--gif" in sys.argv:
    animate_splitting(1, b_max, ZETA, save_path=str(FIG / "p_orbital_split.gif"))
    print("saved animation")
