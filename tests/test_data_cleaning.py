import pandas as pd
import pytest

from src.data.loader import clean_interactions, clean_recipes


def _recipe_df(**overrides):
    base = {
        "id": [1, 2, 3, 4, 5],
        "name": ["rice bowl", None, "stew", "quick cake", "overnight oats"],
        "minutes": [30, 20, 0, 15, 1500],
        "ingredients": [["rice"], ["beef"], ["chicken"], ["flour"], ["oats"]],
        "tags": [[], [], [], [], []],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _interaction_df():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "recipe_id": [10, 20, 10, 30],
            "rating": [0.0, 4.0, 0.0, 5.0],
        }
    )


def test_clean_recipes_drops_null_name():
    df = _recipe_df()
    cleaned = clean_recipes(df)
    assert cleaned["name"].notna().all()
    assert len(cleaned) < len(df)


def test_clean_recipes_drops_zero_minutes():
    df = _recipe_df()
    cleaned = clean_recipes(df)
    assert (cleaned["minutes"] > 0).all()


def test_clean_recipes_drops_over_max_minutes():
    df = _recipe_df()
    cleaned = clean_recipes(df, max_minutes=1440)
    assert (cleaned["minutes"] <= 1440).all()
    # 1500-min row must be gone
    assert 1500 not in cleaned["minutes"].values


def test_clean_recipes_keeps_valid_rows():
    df = _recipe_df()
    cleaned = clean_recipes(df, max_minutes=1440)
    # id=1 (30 min, valid name) must survive
    assert 1 in cleaned["id"].values


def test_clean_recipes_reset_index():
    df = _recipe_df()
    cleaned = clean_recipes(df)
    assert list(cleaned.index) == list(range(len(cleaned)))


def test_clean_interactions_drops_zero_ratings():
    df = _interaction_df()
    cleaned = clean_interactions(df)
    assert (cleaned["rating"] > 0).all()
    assert len(cleaned) == 2


def test_clean_interactions_keeps_nonzero():
    df = _interaction_df()
    cleaned = clean_interactions(df)
    assert set(cleaned["rating"].unique()).issubset({4.0, 5.0})


def test_clean_interactions_no_zeros_noop():
    df = pd.DataFrame({"user_id": [1], "recipe_id": [10], "rating": [5.0]})
    cleaned = clean_interactions(df)
    assert len(cleaned) == 1
