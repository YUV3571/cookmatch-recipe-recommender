import pandas as pd

from src.models.user_profile import UserProfile
from src.recommend.stage2 import Stage2Recommender


def _toy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    recipes = pd.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "name": ["safe bowl", "peanut stew", "veggie pasta", "chicken soup"],
            "ingredients": [
                ["rice", "beans"],
                ["peanut sauce", "noodles"],
                ["pasta", "tomato sauce"],
                ["chicken", "broth"],
            ],
        }
    )
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "recipe_id": [10, 20, 20, 30],
            "u": [0, 0, 1, 1],
            "i": [0, 1, 1, 2],
            "rating": [5.0, 1.0, 4.0, 5.0],
        }
    )
    return recipes, interactions


def test_stage2_excludes_profile_blocked_recipes() -> None:
    recipes, interactions = _toy_data()
    model = Stage2Recommender().fit(recipes, interactions)
    profile = UserProfile(diet="vegetarian", allergens=["nuts"])

    recs = model.recommend(profile, user_id=1, top_n=5)
    ids = [rec.recipe_id for rec in recs]

    assert 20 not in ids
    assert 40 not in ids
    assert all(rec.source == "matrix_factorization" for rec in recs)


def test_stage2_cold_user_uses_popularity_fallback() -> None:
    recipes, interactions = _toy_data()
    model = Stage2Recommender().fit(recipes, interactions)
    profile = UserProfile(diet=None, allergens=[])

    recs = model.recommend(profile, user_id=999, top_n=3)
    assert recs
    assert all(rec.source == "popularity_fallback" for rec in recs)
