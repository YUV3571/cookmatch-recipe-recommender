"""Import this after cloning the repo in Google Colab.

Usage (from repo root):
    import colab_init  # noqa: F401 — adds repo root to sys.path
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ROOT = str(REPO_ROOT)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
