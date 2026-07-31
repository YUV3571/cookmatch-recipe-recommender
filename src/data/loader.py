"""Dataset download and loading helpers."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterable

import kagglehub
import pandas as pd

from config.settings import (
    DATASET_SLUG,
    INTERACTIONS_FILE,
    INTERACTIONS_TEST_FILE,
    INTERACTIONS_TRAIN_FILE,
    INTERACTIONS_VALIDATION_FILE,
    RECIPES_FILE,
)


def get_dataset_path(force_download: bool = False) -> Path:
    """Return local Food.com dataset directory via kagglehub."""
    env_path = os.getenv("FOOD_RS_DATA_PATH")
    if env_path and not force_download:
        path = Path(env_path)
        if path.exists():
            return path

    cache_guess = Path.home() / ".cache/kagglehub/datasets/shuyangli94/food-com-recipes-and-user-interactions/versions/2"
    if cache_guess.exists() and not force_download:
        return cache_guess

    return Path(kagglehub.dataset_download(DATASET_SLUG))


def _parse_list_column(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value.strip().lower()]
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed]
    return [str(value).strip().lower()]


def _normalize_recipe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("ingredients", "tags", "steps"):
        if column in out.columns:
            out[column] = out[column].map(_parse_list_column)
    if "nutrition" in out.columns:
        out["nutrition"] = out["nutrition"].map(_parse_list_column)
    return out


def load_recipes(
    nrows: int | None = None,
    columns: Iterable[str] | None = None,
    dataset_path: Path | None = None,
) -> pd.DataFrame:
    """Load and lightly normalize RAW_recipes.csv."""
    root = dataset_path or get_dataset_path()
    recipe_path = root / RECIPES_FILE
    if not recipe_path.exists():
        raise FileNotFoundError(f"Missing recipes file: {recipe_path}")

    df = pd.read_csv(recipe_path, nrows=nrows, usecols=list(columns) if columns else None)
    return _normalize_recipe_frame(df)


def load_interactions(
    nrows: int | None = None,
    dataset_path: Path | None = None,
) -> pd.DataFrame:
    """Load RAW_interactions.csv."""
    root = dataset_path or get_dataset_path()
    interactions_path = root / INTERACTIONS_FILE
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing interactions file: {interactions_path}")

    return pd.read_csv(interactions_path, nrows=nrows)


def build_eval_recipe_catalog(
    recipes: pd.DataFrame,
    train_interactions: pd.DataFrame,
    held_out: pd.DataFrame,
    max_recipes: int = 30_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Subset recipes for offline eval: all held-out targets plus train recipes up to max_recipes."""
    if "id" not in recipes.columns:
        raise KeyError("recipes must include id column")

    must_include = set(held_out["recipe_id"].astype(int))
    train_ids = train_interactions["recipe_id"].astype(int).unique()
    optional_ids = [int(rid) for rid in train_ids if int(rid) not in must_include]

    remaining = max(0, max_recipes - len(must_include))
    if remaining < len(optional_ids):
        optional_ids = (
            pd.Series(optional_ids)
            .sample(n=remaining, random_state=random_seed, replace=False)
            .astype(int)
            .tolist()
        )

    keep_ids = must_include.union(optional_ids)
    subset = recipes[recipes["id"].astype(int).isin(keep_ids)].copy()
    return subset.reset_index(drop=True)


def filter_interactions_to_catalog(
    interactions: pd.DataFrame,
    recipes: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only interactions whose recipe_id appears in the recipe catalog."""
    catalog_ids = set(recipes["id"].astype(int))
    mask = interactions["recipe_id"].astype(int).isin(catalog_ids)
    return interactions.loc[mask].reset_index(drop=True)


def eval_catalog_coverage(
    recipes: pd.DataFrame,
    held_out: pd.DataFrame,
) -> dict[str, float | int]:
    """Report how many held-out targets exist in the recipe catalog."""
    catalog_ids = set(recipes["id"].astype(int))
    held_out_ids = held_out["recipe_id"].astype(int)
    in_catalog = held_out_ids.isin(catalog_ids)
    total = len(held_out_ids)
    return {
        "held_out_rows": total,
        "held_out_in_catalog": int(in_catalog.sum()),
        "held_out_coverage_pct": round(100.0 * float(in_catalog.mean()), 2) if total else 0.0,
    }


def clean_recipes(
    df: pd.DataFrame,
    max_minutes: int = 1440,
) -> pd.DataFrame:
    """Remove or fix recipe rows that corrupt ranking or time scoring.

    Drops:
    - null name (1 known row in Food.com)
    - minutes == 0 (breaks time-ratio scoring)
    - minutes > max_minutes (default 24 h — 43200-min "recipes" are data noise)
    Returns a reset-index copy with a 'cleaned' provenance column stripped.
    """
    out = df.copy()
    before = len(out)
    out = out[out["name"].notna()]
    out = out[out["minutes"] > 0]
    out = out[out["minutes"] <= max_minutes]
    removed = before - len(out)
    if removed:
        print(f"clean_recipes: removed {removed} rows ({before} → {len(out)})")
    return out.reset_index(drop=True)


def clean_interactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove interactions that corrupt collaborative filtering.

    Drops rating == 0: in Food.com, 0 means 'cooked but did not rate' —
    not a true 0/5 score. Including them flattens SVD factors toward zero
    and degrades Hit@k.
    """
    out = df.copy()
    before = len(out)
    out = out[out["rating"] > 0]
    removed = before - len(out)
    if removed:
        print(f"clean_interactions: removed {removed} zero-rating rows ({before} → {len(out)})")
    return out.reset_index(drop=True)


def load_interaction_split(
    split: str = "train",
    nrows: int | None = None,
    dataset_path: Path | None = None,
) -> pd.DataFrame:
    """Load pre-split interaction file: train, validation, or test."""
    split = split.strip().lower()
    split_files = {
        "train": INTERACTIONS_TRAIN_FILE,
        "validation": INTERACTIONS_VALIDATION_FILE,
        "test": INTERACTIONS_TEST_FILE,
    }
    if split not in split_files:
        raise ValueError(f"Unsupported split: {split!r}. Use: {tuple(split_files)}")

    root = dataset_path or get_dataset_path()
    split_path = root / split_files[split]
    if not split_path.exists():
        raise FileNotFoundError(f"Missing interaction split: {split_path}")

    return pd.read_csv(split_path, nrows=nrows)
