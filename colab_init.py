"""Colab path setup — call bind() at the start of any notebook cell after long runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_DIR = "/content/cookmatch-recipe-recommender"
REPO_ROOT = Path(__file__).resolve().parent


def bind(repo_dir: str | None = None) -> str:
    """Add repo to sys.path and chdir. Safe to call in every cell."""
    root = Path(repo_dir or REPO_DIR)
    if not root.exists():
        raise FileNotFoundError(
            f"Repo not found at {root}. Run notebook cell 1 (GitHub setup) first."
        )
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(root_str)
    return root_str


# auto-bind on import when repo exists (e.g. after cell 1)
if REPO_ROOT.exists() and str(REPO_ROOT) == str(Path(REPO_DIR).resolve()):
    bind(str(REPO_ROOT))
elif Path(REPO_DIR).exists():
    bind(REPO_DIR)
else:
    _root = str(REPO_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)
