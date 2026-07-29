import pytest

from src.filter.ingredient_parser import (
    ingredient_tokens,
    ingredients_to_text,
    keyword_in_ingredients,
    parse_ingredient,
    parse_ingredients,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2 cups milk", "milk"),
        ("unsalted butter", "butter"),
        ("  Olive Oil ", "olive oil"),
        ("1/2 tsp salt", "salt"),
        ("2% low-fat milk", "milk"),
        ("winter squash", "winter squash"),
        ("ground turkey", "turkey"),
    ],
)
def test_parse_ingredient(raw: str, expected: str) -> None:
    assert parse_ingredient(raw) == expected


def test_parse_ingredients_deduplicates() -> None:
    result = parse_ingredients(["Milk", "2 cups milk", "butter", "unsalted butter"])
    assert result == ["milk", "butter"]


def test_ingredients_to_text_joins_normalized_values() -> None:
    text = ingredients_to_text(["2 cups milk", "unsalted butter"])
    assert text == "milk | butter"


def test_keyword_in_ingredients_avoids_nutmeg_false_positive() -> None:
    ingredients = ["nutmeg", "salt", "flour"]
    assert keyword_in_ingredients(ingredients, "nut") is False


def test_keyword_in_ingredients_finds_peanut() -> None:
    ingredients = ["reese's peanut butter cups", "sugar"]
    assert keyword_in_ingredients(ingredients, "peanut") is True


def test_keyword_in_ingredients_finds_multi_word_keyword() -> None:
    ingredients = ["cream cheese", "garlic"]
    assert keyword_in_ingredients(ingredients, "cream cheese") is True


def test_ingredient_tokens() -> None:
    tokens = ingredient_tokens(["2 cups low-fat milk", "olive oil"])
    assert "milk" in tokens
    assert "olive" in tokens
    assert "oil" in tokens
