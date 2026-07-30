import pandas as pd

from src.data.loader import (
    build_eval_recipe_catalog,
    eval_catalog_coverage,
    filter_interactions_to_catalog,
)


def test_build_eval_recipe_catalog_includes_held_out_and_respects_cap() -> None:
    recipes = pd.DataFrame({"id": list(range(1, 101))})
    train = pd.DataFrame({"recipe_id": list(range(1, 91))})
    held_out = pd.DataFrame({"recipe_id": [50, 51, 99]})

    subset = build_eval_recipe_catalog(
        recipes,
        train,
        held_out,
        max_recipes=20,
        random_seed=42,
    )

    assert set(held_out["recipe_id"]).issubset(set(subset["id"]))
    assert len(subset) == 20


def test_filter_interactions_to_catalog() -> None:
    recipes = pd.DataFrame({"id": [1, 2, 3]})
    train = pd.DataFrame({"recipe_id": [1, 2, 99, 3], "rating": [5, 4, 3, 5]})

    filtered = filter_interactions_to_catalog(train, recipes)

    assert filtered["recipe_id"].tolist() == [1, 2, 3]


def test_eval_catalog_coverage() -> None:
    recipes = pd.DataFrame({"id": [1, 2, 3]})
    held_out = pd.DataFrame({"recipe_id": [2, 4, 3]})

    stats = eval_catalog_coverage(recipes, held_out)

    assert stats["held_out_rows"] == 3
    assert stats["held_out_in_catalog"] == 2
    assert stats["held_out_coverage_pct"] == 66.67
