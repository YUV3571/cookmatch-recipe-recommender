"""Colab bootstrap: fetch repo without git if needed."""

from __future__ import annotations

import os
import pathlib
import shutil
import urllib.request
import zipfile

REPO_DIR = "/content/cookmatch-recipe-recommender"
RAW_BASE = "https://raw.githubusercontent.com/YUV3571/cookmatch-recipe-recommender/main"
ZIP_URL = "https://github.com/YUV3571/cookmatch-recipe-recommender/archive/refs/heads/main.zip"


def _download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def ensure_repo() -> str:
    """Download repo zip to Colab and ensure src/data exists."""
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)

    zip_path = pathlib.Path("/content/repo.zip")
    _download(ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")

    extracted = pathlib.Path("/content/cookmatch-recipe-recommender-main")
    if extracted.exists():
        shutil.move(str(extracted), REPO_DIR)

    data_dir = pathlib.Path(REPO_DIR) / "src" / "data"
    loader = data_dir / "loader.py"
    init_file = data_dir / "__init__.py"

    if not loader.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        _download(f"{RAW_BASE}/src/data/loader.py", loader)
        _download(f"{RAW_BASE}/src/data/__init__.py", init_file)

    if not loader.exists():
        raise FileNotFoundError("Failed to bootstrap src/data/loader.py")

    return REPO_DIR


if __name__ == "__main__":
    path = ensure_repo()
    print("Repo ready:", path)
    print("src/data:", os.listdir(os.path.join(path, "src", "data")))
