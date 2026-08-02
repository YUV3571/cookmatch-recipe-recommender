"""Content-based scoring signals for Stage 3 re-ranking."""

from __future__ import annotations

from config.intents import MEAL_INTENT_TAGS
from src.filter.ingredient_parser import parse_ingredient, parse_ingredients


def _stem_word(word: str) -> str:
    w = word.lower().strip()
    if w.endswith("tomatoes"):
        return w[:-2]
    if w.endswith("potatoes"):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and not w.endswith("ches") and not w.endswith("shes"):
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def _stem_text(text: str) -> str:
    tokens = text.lower().split()
    return " ".join([_stem_word(tok) for tok in tokens])


def _pantry_item_matches_recipe_ingredient(pantry_item: str, recipe_ingredient: str) -> bool:
    p_stem = _stem_text(pantry_item)
    r_stem = _stem_text(recipe_ingredient)
    return p_stem in r_stem or r_stem in p_stem


def pantry_match_score(recipe_ingredients: list[str], pantry: list[str]) -> float:
    """Hybrid pantry score: 70% user pantry utilization + 30% recipe coverage."""
    if not pantry:
        return 0.0

    parsed_recipe = parse_ingredients(recipe_ingredients)
    parsed_pantry = parse_ingredients(pantry)
    if not parsed_recipe or not parsed_pantry:
        return 0.0

    recipe_matched_count = 0
    pantry_used_indices = set()

    for r_ingr in parsed_recipe:
        r_stem = _stem_text(r_ingr)
        r_hit = False
        for p_idx, p_ingr in enumerate(parsed_pantry):
            p_stem = _stem_text(p_ingr)
            if p_stem in r_stem or r_stem in p_stem:
                pantry_used_indices.add(p_idx)
                r_hit = True
        if r_hit:
            recipe_matched_count += 1

    pantry_coverage = len(pantry_used_indices) / len(parsed_pantry)
    recipe_coverage = recipe_matched_count / len(parsed_recipe)

    return 0.70 * pantry_coverage + 0.30 * recipe_coverage


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
