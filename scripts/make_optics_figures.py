"""Render every figure and animation of the virtual spectrometer into docs/figures/.

    python scripts/make_optics_figures.py          # static figures only (fast)
    python scripts/make_optics_figures.py --gif    # also the four GIF animations (about a minute)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.optics import figures as F  # noqa: E402
from src.optics.hydrogen_figures import fig_halpha_fan, fig_halpha_regimes  # noqa: E402
from src.optics.instruments import fig_instrument_ladder  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

STATIC = {
    "optics_prism_setup": F.fig_prism_setup,
    "optics_polarisation_filters": F.fig_polarisation_filters,
    "optics_polarisation_ellipses": F.fig_polarisation_ellipses,
    "optics_instrument_ladder": fig_instrument_ladder,
    "halpha_regimes": fig_halpha_regimes,
    "halpha_fan": fig_halpha_fan,
}
ANIMATIONS = {
    "optics_filter_scan": F.animate_filter_scan,
    "optics_zeeman_field_ramp": F.animate_zeeman_field_ramp,
    "optics_polariser_rotation": F.animate_polariser_rotation,
    "optics_view_angle": F.animate_view_angle,
}

for name, make in STATIC.items():
    fig = make()
    fig.savefig(FIG / f"{name}.png", dpi=140, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

if "--gif" in sys.argv:
    for name, make in ANIMATIONS.items():
        t0 = time.time()
        anim = make(save_path=str(FIG / f"{name}.gif"))
        plt.close(anim._fig)
        size_mb = (FIG / f"{name}.gif").stat().st_size / 1e6
        print(f"saved {name}.gif  ({size_mb:.1f} MB, {time.time() - t0:.0f} s)")
