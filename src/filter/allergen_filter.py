"""Hard allergen checks for Stage 1 filtering."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from config.constraints import ALLERGEN_KEYWORDS, SUPPORTED_ALLERGENS
from src.filter.ingredient_parser import keyword_in_ingredients


def _validate_allergens(allergens: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for allergen in allergens:
        key = allergen.strip().lower()
        if key not in SUPPORTED_ALLERGENS:
            raise ValueError(f"Unsupported allergen: {allergen!r}. Supported: {SUPPORTED_ALLERGENS}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def find_allergen_hits(ingredients: Iterable[str], allergen: str) -> list[str]:
    """Return allergen keywords found in a recipe ingredient list."""
    allergen = allergen.strip().lower()
    if allergen not in ALLERGEN_KEYWORDS:
        raise ValueError(f"Unsupported allergen: {allergen!r}")

    hits = [
        keyword
        for keyword in ALLERGEN_KEYWORDS[allergen]
        if keyword_in_ingredients(ingredients, keyword)
    ]
    return hits


def get_allergen_violations(
    ingredients: Iterable[str],
    user_allergens: Iterable[str],
) -> dict[str, list[str]]:
    """Map each user allergen to matched keywords in recipe ingredients."""
    violations: dict[str, list[str]] = {}
    for allergen in _validate_allergens(user_allergens):
        hits = find_allergen_hits(ingredients, allergen)
        if hits:
            violations[allergen] = hits
    return violations


def is_allergen_safe(ingredients: Iterable[str], user_allergens: Iterable[str]) -> bool:
    """True when recipe contains none of the user's declared allergens."""
    return not get_allergen_violations(ingredients, user_allergens)


def filter_recipes_by_allergens(
    recipes: pd.DataFrame,
    user_allergens: Iterable[str],
    ingredient_column: str = "ingredients",
) -> pd.DataFrame:
    """Keep only recipes safe for the user's allergen profile."""
    if ingredient_column not in recipes.columns:
        raise KeyError(f"Missing ingredient column: {ingredient_column}")

    allergens = _validate_allergens(user_allergens)
    if not allergens:
        return recipes.copy()

    mask = recipes[ingredient_column].map(lambda ings: is_allergen_safe(ings, allergens))
    return recipes.loc[mask].copy()


def summarize_allergen_filter(
    recipes: pd.DataFrame,
    user_allergens: Iterable[str],
    ingredient_column: str = "ingredients",
) -> dict[str, int | float]:
    """Basic funnel stats for allergen filtering."""
    total = len(recipes)
    safe = filter_recipes_by_allergens(recipes, user_allergens, ingredient_column)
    safe_count = len(safe)
    return {
        "total_recipes": total,
        "safe_recipes": safe_count,
        "blocked_recipes": total - safe_count,
        "safe_pct": round((safe_count / total) * 100, 2) if total else 0.0,
    }
