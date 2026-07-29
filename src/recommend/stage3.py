"""Stage 3 personalized re-ranking over Stage 2 candidates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.stage3_weights import STAGE3_WEIGHTS
from src.models.session_context import SessionContext
from src.models.user_profile import UserProfile
from src.recommend.content_signals import (
    meal_intent_score,
    normalize_score_map,
    pantry_match_score,
    time_budget_score,
)
from src.recommend.explain import build_explanation
from src.recommend.stage2 import Stage2Recommender


@dataclass(slots=True)
class Stage3Recommendation:
    recipe_id: int
    final_score: float
    source: str
    name: str | None
    cf_score: float
    pantry_score: float
    time_score: float
    intent_score: float
    minutes: int | None
    explanation: str


class Stage3Recommender:
    """Stage 2 scores + pantry/time/intent re-ranking."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(STAGE3_WEIGHTS)
        self.stage2 = Stage2Recommender()
        self.recipe_meta_: dict[int, dict] = {}

    def fit(self, recipes: pd.DataFrame, interactions: pd.DataFrame) -> "Stage3Recommender":
        required = {"id", "name", "ingredients", "minutes", "tags"}
        missing = required - set(recipes.columns)
        if missing:
            raise KeyError(f"recipes missing columns for Stage 3: {sorted(missing)}")

        self.stage2.fit(recipes, interactions)
        self.recipe_meta_ = {
            int(row["id"]): {
                "name": str(row["name"]),
                "ingredients": row["ingredients"],
                "minutes": int(row["minutes"]),
                "tags": row["tags"],
            }
            for _, row in recipes.iterrows()
        }
        return self

    def _active_weights(self, context: SessionContext) -> dict[str, float]:
        active = {"cf": self.weights["cf"]}
        if context.pantry:
            active["pantry"] = self.weights["pantry"]
        if context.max_minutes:
            active["time"] = self.weights["time"]
        if context.meal_intent:
            active["intent"] = self.weights["intent"]

        total = sum(active.values())
        return {key: value / total for key, value in active.items()}

    def _base_scores(
        self,
        profile: UserProfile,
        user_id: int | None,
        candidate_ids: list[int],
    ) -> tuple[dict[int, float], str]:
        if user_id is not None and self.stage2.cf_model.has_user(user_id):
            return self.stage2.cf_model.score(user_id, candidate_ids), "matrix_factorization"
        return self.stage2.popularity_model.score(candidate_ids), "popularity_fallback"

    def recommend(
        self,
        profile: UserProfile,
        context: SessionContext,
        user_id: int | None = None,
        top_n: int = 10,
    ) -> list[Stage3Recommendation]:
        candidate_ids = self.stage2._candidate_ids(profile)
        if not candidate_ids:
            return []

        cf_scores_raw, source = self._base_scores(profile, user_id, candidate_ids)
        cf_scores = normalize_score_map(cf_scores_raw)

        pantry_scores = {
            recipe_id: pantry_match_score(self.recipe_meta_[recipe_id]["ingredients"], context.pantry)
            for recipe_id in candidate_ids
        }
        time_scores = {
            recipe_id: time_budget_score(self.recipe_meta_[recipe_id]["minutes"], context.max_minutes)
            for recipe_id in candidate_ids
        }
        intent_scores = {
            recipe_id: meal_intent_score(self.recipe_meta_[recipe_id]["tags"], context.meal_intent)
            for recipe_id in candidate_ids
        }

        weights = self._active_weights(context)
        ranked: list[Stage3Recommendation] = []

        for recipe_id in candidate_ids:
            meta = self.recipe_meta_[recipe_id]
            final_score = (
                weights.get("cf", 0.0) * cf_scores.get(recipe_id, 0.0)
                + weights.get("pantry", 0.0) * pantry_scores[recipe_id]
                + weights.get("time", 0.0) * time_scores[recipe_id]
                + weights.get("intent", 0.0) * intent_scores[recipe_id]
            )
            ranked.append(
                Stage3Recommendation(
                    recipe_id=recipe_id,
                    final_score=final_score,
                    source=source,
                    name=meta["name"],
                    cf_score=cf_scores.get(recipe_id, 0.0),
                    pantry_score=pantry_scores[recipe_id],
                    time_score=time_scores[recipe_id],
                    intent_score=intent_scores[recipe_id],
                    minutes=meta["minutes"],
                    explanation=build_explanation(
                        source=source,
                        pantry_score=pantry_scores[recipe_id],
                        time_score=time_scores[recipe_id],
                        intent_score=intent_scores[recipe_id],
                        minutes=meta["minutes"],
                        context=context,
                    ),
                )
            )

        ranked.sort(key=lambda item: item.final_score, reverse=True)
        return ranked[:top_n]

    def summarize_stage1_funnel(self, profile: UserProfile) -> dict[str, int | float]:
        return self.stage2.summarize_stage1_funnel(profile)
