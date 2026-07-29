import pandas as pd
import pytest

from src.filter.allergen_filter import (
    filter_recipes_by_allergens,
    find_allergen_hits,
    get_allergen_violations,
    is_allergen_safe,
    summarize_allergen_filter,
)


def test_nuts_detects_peanut_sauce() -> None:
    ingredients = ["peanut sauce", "noodles", "green onion"]
    assert find_allergen_hits(ingredients, "nuts") == ["peanut"]


def test_nuts_ignores_nutmeg() -> None:
    ingredients = ["nutmeg", "flour", "sugar"]
    assert find_allergen_hits(ingredients, "nuts") == []


def test_dairy_detects_ghee() -> None:
    ingredients = ["ghee", "rice", "cumin"]
    assert find_allergen_hits(ingredients, "dairy") == ["ghee"]


def test_gluten_detects_flour() -> None:
    ingredients = ["all-purpose flour", "water", "salt"]
    assert "flour" in find_allergen_hits(ingredients, "gluten")


def test_is_allergen_safe_when_no_overlap() -> None:
    ingredients = ["rice", "black beans", "cumin", "tomatoes"]
    assert is_allergen_safe(ingredients, ["nuts", "dairy", "gluten"]) is True


def test_get_allergen_violations_returns_multiple_allergens() -> None:
    ingredients = ["peanut butter", "milk", "flour"]
    violations = get_allergen_violations(ingredients, ["nuts", "dairy", "gluten"])
    assert set(violations) == {"nuts", "dairy", "gluten"}


def test_filter_recipes_by_allergens() -> None:
    recipes = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["safe bowl", "peanut noodles", "cheese bread"],
            "ingredients": [
                ["rice", "beans", "tomatoes"],
                ["peanut sauce", "noodles"],
                ["flour", "butter", "cheese"],
            ],
        }
    )

    safe = filter_recipes_by_allergens(recipes, ["nuts", "dairy", "gluten"])
    assert safe["id"].tolist() == [1]


def test_summarize_allergen_filter() -> None:
    recipes = pd.DataFrame(
        {
            "id": [1, 2],
            "ingredients": [["rice", "beans"], ["peanut butter", "flour"]],
        }
    )
    summary = summarize_allergen_filter(recipes, ["nuts"])
    assert summary == {
        "total_recipes": 2,
        "safe_recipes": 1,
        "blocked_recipes": 1,
        "safe_pct": 50.0,
    }


def test_unsupported_allergen_raises() -> None:
    with pytest.raises(ValueError):
        is_allergen_safe(["rice"], ["soy"])
