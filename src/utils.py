"""
src/utils.py

Small shared helpers for prepare/train/predict scripts.
Keep this file boring and stable.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict


# ---------- Basic paths ----------
REPO = Path(__file__).resolve().parents[1]  # project/
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"


def dataset_paths(dataset: str) -> Dict[str, Path]:
    """
    Return the stable, canonical paths for a dataset name.

    Notes:
    - We intentionally do NOT include model/preds filenames here because your
      project writes multiple variants (driver/grey, full/d0, localavg, meta, etc.).
      Those names should be owned by the script producing them.
    """
    raw_csv = DATA / f"{dataset}.csv"
    out_dir = OUTPUTS / dataset
    prepared_csv = out_dir / "prepared.csv"
    return {
        "raw_csv": raw_csv,
        "out_dir": out_dir,
        "prepared_csv": prepared_csv,
    }


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn’t exist; return the same path for convenience."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def step(msg: str) -> None:
    """Print a clear progress message."""
    print(f"• {msg}")


def die(msg: str, code: int = 1) -> "None":
    """Print an error and stop execution."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)
