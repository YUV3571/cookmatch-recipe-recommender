"""Bootstrap CookMatch repo inside Google Colab."""

from __future__ import annotations

from colab_init import REPO_DIR, bootstrap_repo, bind, verify_eval_stack


def setup_repo(prefer_git: bool = True) -> str:
    """Download repo to Colab and verify project imports."""
    return bootstrap_repo(REPO_DIR, prefer_git=prefer_git)


def bind_python_path(repo_dir: str | None = None) -> str:
    """Add repo root to sys.path and chdir."""
    return bind(repo_dir)


if __name__ == "__main__":
    path = setup_repo()
    verify_eval_stack()
    print("Repo ready:", path)
