import pandas as pd

from src.recommend.cf_model import MatrixFactorizationModel


def _toy_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "recipe_id": [10, 20, 20, 30, 10, 30],
            "u": [0, 0, 1, 1, 2, 2],
            "i": [0, 1, 1, 2, 0, 2],
            "rating": [5.0, 4.0, 5.0, 3.0, 4.0, 2.0],
        }
    )


def test_cf_scores_known_user() -> None:
    model = MatrixFactorizationModel(n_factors=2).fit(_toy_interactions())
    scores = model.score(1, [10, 20, 30])

    assert len(scores) == 3
    assert scores[20] >= scores[30]


def test_cf_cold_user_falls_back_to_popularity() -> None:
    interactions = _toy_interactions()
    model = MatrixFactorizationModel(n_factors=2).fit(interactions)
    popularity_scores = model._popularity_fallback.score([10, 20, 30])
    scores = model.score(999, [10, 20, 30])

    assert scores == popularity_scores


def test_cf_has_user() -> None:
    model = MatrixFactorizationModel(n_factors=2).fit(_toy_interactions())
    assert model.has_user(1) is True
    assert model.has_user(999) is False
