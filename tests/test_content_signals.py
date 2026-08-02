import pytest

from src.recommend.content_signals import (
    meal_intent_score,
    normalize_score_map,
    pantry_match_score,
    time_budget_score,
)


def test_pantry_match_score_partial_overlap() -> None:
    recipe = ["chicken", "garlic", "onion", "rice"]
    pantry = ["chicken", "garlic", "tomato"]
    assert pantry_match_score(recipe, pantry) == pytest.approx(0.6167, abs=1e-3)


def test_pantry_match_score_empty_pantry() -> None:
    assert pantry_match_score(["rice", "beans"], []) == 0.0


def test_time_budget_score_within_limit() -> None:
    assert time_budget_score(20, 30) == 1.0


def test_time_budget_score_over_limit() -> None:
    score = time_budget_score(60, 30)
    assert 0.0 < score < 1.0


def test_meal_intent_score_match() -> None:
    tags = ["desserts", "easy", "cakes"]
    assert meal_intent_score(tags, "dessert") == 1.0


def test_meal_intent_score_no_match() -> None:
    tags = ["main-dish", "easy"]
    assert meal_intent_score(tags, "dessert") == 0.0


def test_normalize_score_map() -> None:
    normalized = normalize_score_map({1: 2.0, 2: 4.0, 3: 6.0})
    assert normalized[1] == 0.0
    assert normalized[3] == 1.0
