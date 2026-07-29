import pandas as pd
import pytest

from src.filter.profile_filter import (
    filter_recipes_by_profile,
    get_profile_violations,
    get_safe_recipe_ids,
    is_profile_compatible,
    summarize_profile_filter,
)
from src.models.user_profile import UserProfile


def test_profile_blocks_allergen_and_diet_violations() -> None:
    profile = UserProfile(diet="vegan", allergens=["nuts", "gluten"])
    ingredients = ["peanut butter", "milk", "flour"]

    violations = get_profile_violations(ingredients, profile)
    assert violations["allergens"]
    assert violations["diet"]
    assert is_profile_compatible(ingredients, profile) is False


def test_profile_allows_compatible_recipe() -> None:
    profile = UserProfile(diet="vegan", allergens=["nuts", "dairy", "gluten"])
    ingredients = ["rice", "black beans", "tomatoes", "olive oil"]

    assert get_profile_violations(ingredients, profile) == {"allergens": {}, "diet": {}}
    assert is_profile_compatible(ingredients, profile) is True


def test_filter_recipes_by_profile() -> None:
    recipes = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["safe bowl", "peanut stew", "cheese pasta", "chicken soup"],
            "ingredients": [
                ["rice", "beans", "tomatoes"],
                ["peanut sauce", "noodles"],
                ["pasta", "cheese"],
                ["chicken", "broth"],
            ],
        }
    )
    profile = UserProfile(diet="vegetarian", allergens=["nuts"])

    safe = filter_recipes_by_profile(recipes, profile)
    assert safe["id"].tolist() == [1, 3]


def test_get_safe_recipe_ids() -> None:
    recipes = pd.DataFrame(
        {
            "id": [10, 20],
            "ingredients": [["rice", "beans"], ["peanut butter", "flour"]],
        }
    )
    profile = UserProfile(diet="vegan", allergens=["nuts"])

    assert get_safe_recipe_ids(recipes, profile) == [10]


def test_summarize_profile_filter() -> None:
    recipes = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "ingredients": [
                ["rice", "beans"],
                ["peanut butter", "flour"],
                ["pasta", "cheese"],
                ["chicken", "broth"],
            ],
        }
    )
    profile = UserProfile(diet="vegan", allergens=["nuts"])

    summary = summarize_profile_filter(recipes, profile)
    assert summary["total_recipes"] == 4
    assert summary["after_allergen_filter"] == 3
    assert summary["safe_recipes"] == 1
    assert summary["blocked_by_allergens"] == 1
    assert summary["blocked_by_diet"] == 2


def test_user_profile_from_dict() -> None:
    profile = UserProfile.from_dict({"diet": "vegetarian", "allergens": ["dairy"]})
    assert profile.diet == "vegetarian"
    assert profile.allergens == ["dairy"]
