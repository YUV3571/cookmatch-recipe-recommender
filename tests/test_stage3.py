import pandas as pd

from src.models.session_context import SessionContext
from src.models.user_profile import UserProfile
from src.recommend.stage3 import Stage3Recommender


def _toy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    recipes = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "name": ["quick rice bowl", "slow nut stew", "quick dessert"],
            "minutes": [15, 120, 20],
            "tags": [["main-dish", "easy"], ["main-dish"], ["desserts", "easy"]],
            "ingredients": [
                ["rice", "beans", "garlic"],
                ["peanut sauce", "noodles"],
                ["flour", "sugar", "cocoa"],
            ],
        }
    )
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "recipe_id": [10, 20, 30, 10, 30],
            "u": [0, 0, 0, 1, 1],
            "i": [0, 1, 2, 0, 2],
            "rating": [4.0, 4.0, 4.0, 4.0, 4.0],
        }
    )
    return recipes, interactions


def test_stage3_prefers_pantry_and_time_for_context() -> None:
    recipes, interactions = _toy_data()
    weights = {"cf": 0.10, "pantry": 0.60, "time": 0.20, "intent": 0.10}
    model = Stage3Recommender(weights=weights).fit(recipes, interactions)
    profile = UserProfile(diet="vegetarian", allergens=["nuts"])
    context = SessionContext(pantry=["rice", "beans"], max_minutes=20, meal_intent="main")

    recs = model.recommend(profile, context, user_id=1, top_n=3)
    assert recs
    assert recs[0].recipe_id == 10
    assert recs[0].pantry_score > 0
    assert "pantry match" in recs[0].explanation


def test_stage3_meal_intent_boosts_dessert() -> None:
    recipes, interactions = _toy_data()
    model = Stage3Recommender().fit(recipes, interactions)
    profile = UserProfile(diet=None, allergens=[])
    context = SessionContext(meal_intent="dessert")

    recs = model.recommend(profile, context, user_id=1, top_n=1)
    assert recs[0].recipe_id == 30
    assert recs[0].intent_score == 1.0
