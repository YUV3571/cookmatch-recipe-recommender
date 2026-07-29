import pandas as pd

from src.recommend.popularity import PopularityModel


def test_popularity_prefers_highly_rated_recipe() -> None:
    interactions = pd.DataFrame(
        {
            "recipe_id": [1, 1, 1, 2, 2],
            "rating": [5, 5, 4, 2, 1],
        }
    )
    model = PopularityModel(min_prior=1).fit(interactions)
    scores = model.score([1, 2])

    assert scores[1] > scores[2]


def test_popularity_unknown_recipe_uses_global_mean() -> None:
    interactions = pd.DataFrame({"recipe_id": [1], "rating": [4.0]})
    model = PopularityModel(min_prior=1).fit(interactions)

    assert model.score([999])[999] == model.global_mean
