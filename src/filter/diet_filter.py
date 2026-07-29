"""Hard diet checks for Stage 1 filtering."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from config.constraints import (
    ANIMAL_DERIVED_KEYWORDS,
    FISH_KEYWORDS,
    MEAT_KEYWORDS,
    SUPPORTED_DIETS,
)
from src.filter.ingredient_parser import keyword_in_ingredients

DIET_KEYWORDS = {
    "vegetarian": tuple(MEAT_KEYWORDS) + tuple(FISH_KEYWORDS),
    "vegan": tuple(MEAT_KEYWORDS) + tuple(FISH_KEYWORDS) + tuple(ANIMAL_DERIVED_KEYWORDS),
}


def _validate_diet(diet: str | None) -> str | None:
    if diet is None:
        return None
    key = diet.strip().lower()
    if key in ("none", "omnivore", ""):
        return None
    if key not in SUPPORTED_DIETS:
        raise ValueError(f"Unsupported diet: {diet!r}. Supported: {SUPPORTED_DIETS}")
    return key


def find_diet_hits(ingredients: Iterable[str], diet: str) -> list[str]:
    """Return blocked keywords found for the given diet."""
    diet = _validate_diet(diet)
    if diet is None:
        return []

    hits = [
        keyword
        for keyword in DIET_KEYWORDS[diet]
        if keyword_in_ingredients(ingredients, keyword)
    ]
    return hits


def get_diet_violations(ingredients: Iterable[str], diet: str | None) -> dict[str, list[str]]:
    """Return diet violations with matched keywords."""
    diet = _validate_diet(diet)
    if diet is None:
        return {}

    hits = find_diet_hits(ingredients, diet)
    if not hits:
        return {}
    return {diet: hits}


def is_diet_compatible(ingredients: Iterable[str], diet: str | None) -> bool:
    """True when recipe satisfies the user's diet preference."""
    return not get_diet_violations(ingredients, diet)


def filter_recipes_by_diet(
    recipes: pd.DataFrame,
    diet: str | None,
    ingredient_column: str = "ingredients",
) -> pd.DataFrame:
    """Keep only recipes compatible with the user's diet."""
    if ingredient_column not in recipes.columns:
        raise KeyError(f"Missing ingredient column: {ingredient_column}")

    validated = _validate_diet(diet)
    if validated is None:
        return recipes.copy()

    mask = recipes[ingredient_column].map(lambda ings: is_diet_compatible(ings, validated))
    return recipes.loc[mask].copy()


def summarize_diet_filter(
    recipes: pd.DataFrame,
    diet: str | None,
    ingredient_column: str = "ingredients",
) -> dict[str, int | float]:
    """Basic funnel stats for diet filtering."""
    total = len(recipes)
    compatible = filter_recipes_by_diet(recipes, diet, ingredient_column)
    compatible_count = len(compatible)
    return {
        "total_recipes": total,
        "compatible_recipes": compatible_count,
        "blocked_recipes": total - compatible_count,
        "compatible_pct": round((compatible_count / total) * 100, 2) if total else 0.0,
    }
