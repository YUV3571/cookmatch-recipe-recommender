"""Content-based scoring signals for Stage 3 re-ranking."""

from __future__ import annotations

from config.intents import MEAL_INTENT_TAGS
from src.filter.ingredient_parser import parse_ingredient, parse_ingredients


def _pantry_item_matches_recipe_ingredient(pantry_item: str, recipe_ingredient: str) -> bool:
    return pantry_item in recipe_ingredient or recipe_ingredient in pantry_item


def pantry_match_score(recipe_ingredients: list[str], pantry: list[str]) -> float:
    """Fraction of recipe ingredients covered by pantry items."""
    if not pantry:
        return 0.0

    parsed_recipe = parse_ingredients(recipe_ingredients)
    parsed_pantry = parse_ingredients(pantry)
    if not parsed_recipe or not parsed_pantry:
        return 0.0

    matched = 0
    for recipe_ingredient in parsed_recipe:
        if any(_pantry_item_matches_recipe_ingredient(item, recipe_ingredient) for item in parsed_pantry):
            matched += 1

    return matched / len(parsed_recipe)


def time_budget_score(minutes: int, max_minutes: int | None) -> float:
    """Higher score for recipes within the user's time budget."""
    if max_minutes is None or max_minutes <= 0:
        return 0.0
    if minutes <= max_minutes:
        return 1.0
    return max(0.0, max_minutes / minutes)


def meal_intent_score(tags: list[str], meal_intent: str | None) -> float:
    """1.0 when recipe tags include the requested meal intent."""
    if not meal_intent:
        return 0.0

    target_tag = MEAL_INTENT_TAGS.get(meal_intent.strip().lower())
    if not target_tag:
        return 0.0

    normalized_tags = {parse_ingredient(tag) for tag in tags}
    return 1.0 if target_tag in normalized_tags else 0.0


def normalize_score_map(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize scores to [0, 1] within a candidate set."""
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}
