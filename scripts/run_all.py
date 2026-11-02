"""One command to rebuild everything, on any operating system.

    python scripts/run_all.py            # tests, figures, animations, notebooks, PDF
    python scripts/run_all.py --fast     # skip the GIF animations
    python scripts/run_all.py --no-pdf   # skip LaTeX (if you do not have it installed)

Steps run in order and the script stops at the first failure.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
fast = "--fast" in sys.argv
gif = [] if fast else ["--gif"]

STEPS = [
    ("tests", [PY, "-m", "pytest", "-q"], ROOT),
    ("physics figures", [PY, "scripts/make_figures.py", *gif], ROOT),
    ("optics figures", [PY, "scripts/make_optics_figures.py", *gif], ROOT),
    ("notebooks", [PY, "scripts/make_notebooks.py"], ROOT),
    ("execute notebooks", [PY, "scripts/execute_notebooks.py"], ROOT),
]
if "--no-pdf" not in sys.argv:
    if shutil.which("latexmk"):
        STEPS.append(("PDF report", ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], ROOT / "docs"))
    else:
        print("latexmk not found: skipping the PDF (install a TeX distribution, or use --no-pdf)")

for name, cmd, cwd in STEPS:
    print(f"\n=== {name} ===", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"step failed: {name}")
print("\nAll steps finished.")
