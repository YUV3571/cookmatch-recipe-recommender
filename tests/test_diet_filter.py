import pandas as pd
import pytest

from src.filter.diet_filter import (
    filter_recipes_by_diet,
    find_diet_hits,
    get_diet_violations,
    is_diet_compatible,
    summarize_diet_filter,
)


def test_vegetarian_blocks_chicken() -> None:
    ingredients = ["chicken broth", "noodles", "carrots"]
    assert is_diet_compatible(ingredients, "vegetarian") is False
    assert "chicken" in find_diet_hits(ingredients, "vegetarian")


def test_vegetarian_allows_cheese() -> None:
    ingredients = ["pasta", "tomato sauce", "mozzarella"]
    assert is_diet_compatible(ingredients, "vegetarian") is True


def test_vegan_blocks_eggs_and_dairy() -> None:
    ingredients = ["flour", "milk", "eggs", "sugar"]
    violations = get_diet_violations(ingredients, "vegan")
    assert violations["vegan"]
    assert is_diet_compatible(ingredients, "vegan") is False


def test_vegan_allows_plant_recipe() -> None:
    ingredients = ["rice", "black beans", "tomatoes", "olive oil"]
    assert is_diet_compatible(ingredients, "vegan") is True


def test_no_diet_allows_everything() -> None:
    ingredients = ["chicken", "milk", "eggs"]
    assert is_diet_compatible(ingredients, None) is True
    assert get_diet_violations(ingredients, None) == {}


def test_filter_recipes_by_diet() -> None:
    recipes = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "ingredients": [
                ["rice", "beans"],
                ["chicken", "garlic"],
                ["pasta", "cheese"],
            ],
        }
    )

    vegetarian = filter_recipes_by_diet(recipes, "vegetarian")
    vegan = filter_recipes_by_diet(recipes, "vegan")

    assert vegetarian["id"].tolist() == [1, 3]
    assert vegan["id"].tolist() == [1]


def test_summarize_diet_filter() -> None:
    recipes = pd.DataFrame(
        {
            "id": [1, 2],
            "ingredients": [["rice", "beans"], ["beef", "onion"]],
        }
    )
    summary = summarize_diet_filter(recipes, "vegetarian")
    assert summary == {
        "total_recipes": 2,
        "compatible_recipes": 1,
        "blocked_recipes": 1,
        "compatible_pct": 50.0,
    }


def test_unsupported_diet_raises() -> None:
    with pytest.raises(ValueError):
        is_diet_compatible(["rice"], "pescatarian")
