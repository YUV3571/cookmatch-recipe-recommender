"""Bootstrap CookMatch repo inside Google Colab."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import urllib.request
import zipfile

REPO_DIR = "/content/cookmatch-recipe-recommender"
REPO_URL = "https://github.com/YUV3571/cookmatch-recipe-recommender.git"
RAW_BASE = "https://raw.githubusercontent.com/YUV3571/cookmatch-recipe-recommender/main"
ZIP_URL = "https://github.com/YUV3571/cookmatch-recipe-recommender/archive/refs/heads/main.zip"

REQUIRED_PATHS = (
    "src/data/loader.py",
    "src/recommend/stage3.py",
    "config/settings.py",
    "colab_init.py",
)


def _download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _verify(repo_path: pathlib.Path) -> None:
    missing = [rel for rel in REQUIRED_PATHS if not (repo_path / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Repo incomplete. Missing: {missing}")


def _patch_src_data(repo_path: pathlib.Path) -> None:
    data_dir = repo_path / "src" / "data"
    loader = data_dir / "loader.py"
    init_file = data_dir / "__init__.py"
    if loader.exists() and init_file.exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    _download(f"{RAW_BASE}/src/data/loader.py", loader)
    _download(f"{RAW_BASE}/src/data/__init__.py", init_file)


def _clone_with_git(repo_path: pathlib.Path) -> bool:
    if repo_path.exists():
        shutil.rmtree(repo_path)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and repo_path.exists()


def _clone_with_zip(repo_path: pathlib.Path) -> None:
    if repo_path.exists():
        shutil.rmtree(repo_path)

    zip_path = pathlib.Path("/content/repo.zip")
    _download(ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")

    extracted = pathlib.Path("/content/cookmatch-recipe-recommender-main")
    if not extracted.exists():
        raise FileNotFoundError("GitHub zip download did not contain expected folder")

    shutil.move(str(extracted), str(repo_path))


def setup_repo(prefer_git: bool = True) -> str:
    """Download repo to Colab and verify project imports."""
    repo_path = pathlib.Path(REPO_DIR)
    ok = prefer_git and _clone_with_git(repo_path)
    if not ok:
        _clone_with_zip(repo_path)

    _patch_src_data(repo_path)
    _verify(repo_path)

    return str(repo_path.resolve())


def bind_python_path(repo_dir: str | None = None) -> str:
    """Add repo root to sys.path and optionally chdir."""
    import sys

    root = repo_dir or REPO_DIR
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    return root


if __name__ == "__main__":
    path = setup_repo()
    bind_python_path(path)
    print("Repo ready:", path)
    print("Verified:", REQUIRED_PATHS)
