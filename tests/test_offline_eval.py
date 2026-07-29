import pandas as pd

from src.eval.offline_eval import run_ablation
from src.models.user_profile import UserProfile


def _toy_eval_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recipes = pd.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "name": ["rice bowl", "peanut stew", "quick dessert", "chicken soup"],
            "minutes": [15, 120, 20, 25],
            "tags": [["main-dish"], ["main-dish"], ["desserts"], ["main-dish"]],
            "ingredients": [
                ["rice", "beans"],
                ["peanut sauce"],
                ["sugar", "flour"],
                ["chicken", "broth"],
            ],
        }
    )
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "recipe_id": [10, 20, 20, 30, 10, 30],
            "u": [0, 0, 1, 1, 2, 2],
            "i": [0, 1, 1, 2, 0, 2],
            "rating": [5.0, 1.0, 4.0, 5.0, 4.0, 5.0],
        }
    )
    held_out = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "recipe_id": [30, 10, 20],
            "rating": [4.0, 4.0, 4.0],
        }
    )
    return recipes, train, held_out


def test_run_ablation_returns_expected_scenarios() -> None:
    recipes, train, held_out = _toy_eval_data()
    results = run_ablation(
        recipes=recipes,
        train_interactions=train,
        held_out=held_out,
        user_sample_size=3,
        k=2,
    )

    assert len(results) == 7
    assert "stage2_mf_open" in results["scenario"].values
    assert float(results.loc[results["scenario"] == "stage3_strict_profile", "constraint_violation_rate"].iloc[0]) == 0.0
