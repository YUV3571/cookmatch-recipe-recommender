"""Unified Stage 1 profile filter API."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.filter.allergen_filter import (
    filter_recipes_by_allergens,
    get_allergen_violations,
    is_allergen_safe,
)
from src.filter.diet_filter import (
    filter_recipes_by_diet,
    get_diet_violations,
    is_diet_compatible,
)
from src.models.user_profile import UserProfile


def get_profile_violations(
    ingredients: Iterable[str],
    profile: UserProfile,
) -> dict[str, dict[str, list[str]]]:
    """Return structured allergen and diet violations for one recipe."""
    return {
        "allergens": get_allergen_violations(ingredients, profile.allergens),
        "diet": get_diet_violations(ingredients, profile.diet),
    }


def is_profile_compatible(ingredients: Iterable[str], profile: UserProfile) -> bool:
    """True when recipe passes both allergen and diet constraints."""
    return is_allergen_safe(ingredients, profile.allergens) and is_diet_compatible(
        ingredients, profile.diet
    )


def filter_recipes_by_profile(
    recipes: pd.DataFrame,
    profile: UserProfile,
    ingredient_column: str = "ingredients",
) -> pd.DataFrame:
    """Keep recipes compatible with the full user profile."""
    if ingredient_column not in recipes.columns:
        raise KeyError(f"Missing ingredient column: {ingredient_column}")

    filtered = filter_recipes_by_allergens(recipes, profile.allergens, ingredient_column)
    filtered = filter_recipes_by_diet(filtered, profile.diet, ingredient_column)
    return filtered.reset_index(drop=True)


def get_safe_recipe_ids(
    recipes: pd.DataFrame,
    profile: UserProfile,
    id_column: str = "id",
    ingredient_column: str = "ingredients",
) -> list[int]:
    """Return safe recipe IDs for downstream recommendation stages."""
    safe = filter_recipes_by_profile(recipes, profile, ingredient_column)
    if id_column not in safe.columns:
        raise KeyError(f"Missing id column: {id_column}")
    return safe[id_column].tolist()


def summarize_profile_filter(
    recipes: pd.DataFrame,
    profile: UserProfile,
    ingredient_column: str = "ingredients",
) -> dict[str, int | float]:
    """Funnel stats for profile-based hard filtering."""
    total = len(recipes)
    allergen_safe = filter_recipes_by_allergens(recipes, profile.allergens, ingredient_column)
    final_safe = filter_recipes_by_diet(allergen_safe, profile.diet, ingredient_column)

    allergen_safe_count = len(allergen_safe)
    final_count = len(final_safe)

    return {
        "total_recipes": total,
        "after_allergen_filter": allergen_safe_count,
        "safe_recipes": final_count,
        "blocked_recipes": total - final_count,
        "safe_pct": round((final_count / total) * 100, 2) if total else 0.0,
        "blocked_by_allergens": total - allergen_safe_count,
        "blocked_by_diet": allergen_safe_count - final_count,
    }
