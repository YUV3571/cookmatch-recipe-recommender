"""Stage 2 orchestration: profile-safe pool + CF/popularity ranking."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.filter.profile_filter import filter_recipes_by_profile, summarize_profile_filter
from src.models.user_profile import UserProfile
from src.recommend.cf_model import MatrixFactorizationModel
from src.recommend.popularity import PopularityModel


@dataclass(slots=True)
class Recommendation:
    recipe_id: int
    score: float
    source: str
    name: str | None = None


class Stage2Recommender:
    """Combine Stage 1 hard filtering with Stage 2 collaborative scoring."""

    def __init__(self) -> None:
        self.cf_model = MatrixFactorizationModel()
        self.popularity_model = PopularityModel()
        self.recipes_: pd.DataFrame | None = None
        self.recipe_names_: dict[int, str] = {}
        self._candidate_cache: dict[tuple[str | None, tuple[str, ...]], list[int]] = {}

    def fit(self, recipes: pd.DataFrame, interactions: pd.DataFrame) -> "Stage2Recommender":
        if "id" not in recipes.columns:
            raise KeyError("recipes must include id column")

        self.recipes_ = recipes.copy()
        self.recipe_names_ = {
            int(row["id"]): str(row["name"]) for _, row in recipes[["id", "name"]].iterrows()
        }
        self._candidate_cache = {}
        self.cf_model.fit(interactions)
        self.popularity_model.fit(interactions)
        return self

    def _profile_cache_key(self, profile: UserProfile) -> tuple[str | None, tuple[str, ...]]:
        return profile.diet, tuple(sorted(profile.allergens))

    def _candidate_ids(self, profile: UserProfile) -> list[int]:
        if self.recipes_ is None:
            raise RuntimeError("Stage2Recommender must be fit before recommendation")

        cache_key = self._profile_cache_key(profile)
        if cache_key not in self._candidate_cache:
            safe = filter_recipes_by_profile(self.recipes_, profile)
            self._candidate_cache[cache_key] = safe["id"].astype(int).tolist()
        return self._candidate_cache[cache_key]

    def recommend(
        self,
        profile: UserProfile,
        user_id: int | None,
        top_n: int = 10,
    ) -> list[Recommendation]:
        candidate_ids = self._candidate_ids(profile)
        if not candidate_ids:
            return []

        if user_id is not None and self.cf_model.has_user(user_id):
            ranked = self.cf_model.rank(user_id, candidate_ids, top_n=top_n)
            source = "matrix_factorization"
        else:
            ranked = self.popularity_model.rank(candidate_ids, top_n=top_n)
            source = "popularity_fallback"

        return [
            Recommendation(
                recipe_id=recipe_id,
                score=score,
                source=source,
                name=self.recipe_names_.get(recipe_id),
            )
            for recipe_id, score in ranked
        ]

    def summarize_stage1_funnel(self, profile: UserProfile) -> dict[str, int | float]:
        if self.recipes_ is None:
            raise RuntimeError("Stage2Recommender must be fit before summarizing")
        return summarize_profile_filter(self.recipes_, profile)
