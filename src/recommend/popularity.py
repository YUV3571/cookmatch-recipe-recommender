"""Popularity baseline for cold-start and fallback scoring."""

from __future__ import annotations

import pandas as pd

from config.settings import POPULARITY_MIN_PRIOR


class PopularityModel:
    """Bayesian-smoothed mean rating per recipe."""

    def __init__(self, min_prior: int = POPULARITY_MIN_PRIOR) -> None:
        self.min_prior = min_prior
        self.global_mean: float = 0.0
        self.scores_: dict[int, float] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularityModel":
        if "recipe_id" not in interactions.columns or "rating" not in interactions.columns:
            raise KeyError("interactions must include recipe_id and rating columns")

        self.global_mean = float(interactions["rating"].mean())
        stats = interactions.groupby("recipe_id")["rating"].agg(["mean", "count"])

        for recipe_id, row in stats.iterrows():
            count = float(row["count"])
            mean = float(row["mean"])
            smoothed = (count * mean + self.min_prior * self.global_mean) / (count + self.min_prior)
            self.scores_[int(recipe_id)] = smoothed

        return self

    def score(self, recipe_ids: list[int]) -> dict[int, float]:
        fallback = self.global_mean
        return {int(recipe_id): self.scores_.get(int(recipe_id), fallback) for recipe_id in recipe_ids}

    def rank(self, recipe_ids: list[int], top_n: int = 10) -> list[tuple[int, float]]:
        scored = self.score(recipe_ids)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_n]
