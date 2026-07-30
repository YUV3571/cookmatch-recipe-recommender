import pandas as pd

from src.data.loader import build_eval_recipe_catalog


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
