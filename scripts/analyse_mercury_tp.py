"""Analyse the mercury Zeeman teaching lab from a CSV of doublets.

usage: python scripts/analyse_mercury_tp.py data/mercury_tp_measurements_TEMPLATE.csv [B_gauss] [T_kelvin] [sigma_B_gauss]
The CSV has two columns, p1_mmhg and p2_mmhg: the pressures of the two neighbouring peaks of each doublet read on the recording.
Lines starting with '#' are comments. Defaults: B = 2340 G (1 A), T = 300 K, sigma_B = 20 G.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mercury import analyse_pairs  # noqa: E402


def main(argv):
    path = argv[1]
    b = float(argv[2]) if len(argv) > 2 else 2340.0
    t = float(argv[3]) if len(argv) > 3 else 300.0
    sb = float(argv[4]) if len(argv) > 4 else 20.0
    df = pd.read_csv(path, comment="#").dropna()
    if df.empty:
        raise SystemExit("no complete rows in the CSV: fill in p1_mmhg and p2_mmhg")
    r = analyse_pairs(list(zip(df.p1_mmhg, df.p2_mmhg)), b, t, sb)
    print(r["table"].to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(f"\nB = {b:g} G, T = {t:g} K, {len(df)} doublets")
    print(f"mu_B/hc = {r['mean']:.4e} +- {r['uncertainty']:.1e} cm^-1/G   (scatter of readings {r['std']:.1e}, standard error {r['sem']:.1e})")
    print(f"CODATA  = {r['codata']:.4e} cm^-1/G   deviation {100 * r['relative_deviation']:+.2f} %  ({r['n_sigma']:+.1f} sigma)")


if __name__ == "__main__":
    main(sys.argv)
