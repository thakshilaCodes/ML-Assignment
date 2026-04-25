"""
Run the XGBoost training notebook as the single training pipeline.

`python models/xgboost/train.py` executes `notebooks/xgboost/xgboost_training.ipynb`
and writes an executed copy to `outputs/xgboost/metrics/xgboost_training_executed.ipynb`.
The notebook itself is responsible for saving model/metrics/prediction artifacts.
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import traceback


ALGO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALGO_DIR.parent.parent
NOTEBOOK_IN = PROJECT_ROOT / "notebooks" / "xgboost" / "xgboost_training.ipynb"
NOTEBOOK_OUT = (
    PROJECT_ROOT / "outputs" / "xgboost" / "metrics" / "xgboost_training_executed.ipynb"
)


def _strip_notebook_magics(code: str) -> str:
    """Remove IPython-only line magics so plain Python can execute notebook cells."""
    kept: list[str] = []
    for line in code.splitlines():
        s = line.lstrip()
        if s.startswith("%") or s.startswith("!"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _run_notebook_code_cells(notebook_path: Path, *, cwd: Path) -> None:
    raw = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = raw.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Invalid notebook format: {notebook_path}")

    namespace: dict[str, object] = {"__name__": "__main__"}
    original_cwd = Path.cwd()
    try:
        # Keep notebook relative paths consistent with normal interactive usage.
        import os

        os.chdir(cwd)
        for idx, cell in enumerate(cells, start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            code = _strip_notebook_magics(source).strip()
            if not code:
                continue
            print(f"[train.py] Running notebook code cell {idx}...")
            try:
                exec(compile(code, f"{notebook_path}#cell{idx}", "exec"), namespace, namespace)
            except Exception as exc:
                raise RuntimeError(
                    f"Notebook execution failed at code cell {idx}: {exc}"
                ) from exc
    finally:
        import os

        os.chdir(original_cwd)


def main() -> None:
    if not NOTEBOOK_IN.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_IN}")

    NOTEBOOK_OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Executing notebook code cells:", NOTEBOOK_IN)
    try:
        _run_notebook_code_cells(NOTEBOOK_IN, cwd=PROJECT_ROOT)
    except Exception:
        traceback.print_exc()
        raise

    shutil.copy2(NOTEBOOK_IN, NOTEBOOK_OUT)
    print("Saved notebook snapshot to:", NOTEBOOK_OUT)
    print("Training artifacts were written by notebook cells under outputs/xgboost/...")


if __name__ == "__main__":
    main()
