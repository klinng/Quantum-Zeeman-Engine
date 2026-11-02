"""Minimal notebook runner: executes every code cell and stores printed text and figures in the
.ipynb, so the notebooks show their results on GitHub without needing Jupyter installed.

    python scripts/execute_notebooks.py

(With Jupyter you can instead run: jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb)
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _capture_figures(images):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        images.append(base64.b64encode(buf.getvalue()).decode())
        plt.close(fig)


def run_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    old_cwd = os.getcwd()
    os.chdir(path.parent)                      # notebooks do sys.path.insert(0, '..')
    ok = True
    count = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += 1
            images, out = [], io.StringIO()
            plt.show = lambda *a, **k: _capture_figures(images)      # what plt.show() does in a notebook
            outputs = []
            try:
                with contextlib.redirect_stdout(out):
                    exec(compile("".join(cell["source"]), f"{path.name}[cell {count}]", "exec"), ns)
            except Exception:
                ok = False
                outputs.append({"output_type": "error", "ename": "Error", "evalue": "see traceback",
                                "traceback": traceback.format_exc().splitlines()})
            _capture_figures(images)             # figures made but never shown
            if out.getvalue():
                outputs.insert(0, {"output_type": "stream", "name": "stdout",
                                   "text": out.getvalue().splitlines(keepends=True)})
            for img in images:
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": img, "text/plain": ["<Figure>"]}})
            cell["outputs"], cell["execution_count"] = outputs, count
    finally:
        os.chdir(old_cwd)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return ok


if __name__ == "__main__":
    sys.path.insert(0, str(NB_DIR.parent))
    results = {p.name: run_notebook(p) for p in sorted(NB_DIR.glob("*.ipynb"))}
    for name, ok in results.items():
        print(("OK    " if ok else "ERROR ") + name)
    sys.exit(0 if all(results.values()) else 1)
