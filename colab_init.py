"""Colab bootstrap: clone repo, sync critical files, purge stale imports."""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_DIR = "/content/cookmatch-recipe-recommender"
REPO_URL = "https://github.com/YUV3571/cookmatch-recipe-recommender.git"
RAW_BASE = "https://raw.githubusercontent.com/YUV3571/cookmatch-recipe-recommender/main"
ZIP_URL = "https://github.com/YUV3571/cookmatch-recipe-recommender/archive/refs/heads/main.zip"

# Always re-download these from main after clone (avoids stale clone/cache)
SYNC_PATHS = (
    "colab_init.py",
    "config/settings.py",
    "src/data/loader.py",
    "src/eval/offline_eval.py",
    "src/recommend/stage3.py",
)


def purge_cached_modules() -> None:
    """Drop cached project modules so a fresh clone is actually imported."""
    for name in list(sys.modules):
        if name in {"colab_init", "config"} or name.startswith("src."):
            del sys.modules[name]


def reset_sys_path(repo_dir: str) -> None:
    """Prefer the cloned repo on sys.path over any stale copies."""
    repo_posix = Path(repo_dir).resolve().as_posix()
    sys.path[:] = [
        entry
        for entry in sys.path
        if "cookmatch-recipe-recommender" not in Path(entry).as_posix()
    ]
    if repo_posix not in sys.path:
        sys.path.insert(0, repo_posix)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def sync_files_from_github(repo_dir: str) -> None:
    """Overwrite critical files from GitHub main."""
    root = Path(repo_dir)
    for rel in SYNC_PATHS:
        _download(f"{RAW_BASE}/{rel}", root / rel)


def _clone_with_git(repo_path: Path) -> bool:
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and repo_path.exists()


def _clone_with_zip(repo_path: Path) -> None:
    zip_path = Path("/content/repo.zip")
    _download(ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")
    extracted = Path("/content/cookmatch-recipe-recommender-main")
    if not extracted.exists():
        raise FileNotFoundError("GitHub zip download did not contain expected folder")
    shutil.move(str(extracted), str(repo_path))


def download_repo(repo_dir: str = REPO_DIR, prefer_git: bool = True) -> Path:
    """Clone or zip-download the repo into Colab."""
    os.chdir("/content")
    repo_path = Path(repo_dir)
    if repo_path.exists():
        shutil.rmtree(repo_path)

    if prefer_git and _clone_with_git(repo_path):
        print("Connected via git clone")
    else:
        print("git clone failed — using zip fallback (normal on Colab)")
        _clone_with_zip(repo_path)

    return repo_path.resolve()


def bootstrap_repo(repo_dir: str = REPO_DIR, prefer_git: bool = True) -> str:
    """Full Colab setup: clone, sync critical files, purge imports, bind path."""
    repo_path = download_repo(repo_dir, prefer_git=prefer_git)
    sync_files_from_github(str(repo_path))
    return bind(str(repo_path))


def bind(repo_dir: str | None = None) -> str:
    """Bind Colab runtime to repo path (safe to call in every notebook cell)."""
    root = Path(repo_dir or REPO_DIR)
    if not root.exists():
        raise FileNotFoundError(
            f"Repo not found at {root}. Run notebook cell 1 (bootstrap_repo) first."
        )
    purge_cached_modules()
    root_str = str(root.resolve())
    reset_sys_path(root_str)
    os.chdir(root_str)
    return root_str


def verify_eval_stack() -> None:
    """Confirm Stage 3 eval fixes are importable."""
    from src.recommend.stage3 import Stage3Recommender

    params = inspect.signature(Stage3Recommender.recommend).parameters
    if "pin_recipe_ids" not in params:
        raise RuntimeError(
            "Stale CookMatch code still loaded. Runtime → Restart session, then re-run cell 1."
        )


def load_bootstrap_module():
    """Fetch latest colab_init.py from GitHub main before cloning (entrypoint for cell 1)."""
    entry = Path("/content/_cookmatch_colab_init.py")
    _download(f"{RAW_BASE}/colab_init.py", entry)
    purge_cached_modules()

    import importlib.util

    spec = importlib.util.spec_from_file_location("colab_init", entry)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load colab_init bootstrap module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["colab_init"] = module
    spec.loader.exec_module(module)
    return module
