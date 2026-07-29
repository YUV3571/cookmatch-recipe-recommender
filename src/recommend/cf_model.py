"""Matrix factorization recommender (truncated SVD) for Stage 2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from config.settings import CF_FACTORS
from src.recommend.popularity import PopularityModel


class MatrixFactorizationModel:
    """User-item MF model trained with scipy.sparse.linalg.svds."""

    def __init__(self, n_factors: int = CF_FACTORS) -> None:
        self.n_factors = n_factors
        self.global_mean: float = 0.0
        self.user_ids_: np.ndarray | None = None
        self.recipe_ids_: np.ndarray | None = None
        self.user_index_: dict[int, int] = {}
        self.recipe_index_: dict[int, int] = {}
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self._popularity_fallback = PopularityModel()

    def fit(self, interactions: pd.DataFrame) -> "MatrixFactorizationModel":
        required = {"u", "i", "recipe_id", "user_id", "rating"}
        missing = required - set(interactions.columns)
        if missing:
            raise KeyError(f"interactions missing columns: {sorted(missing)}")

        self.global_mean = float(interactions["rating"].mean())
        self._popularity_fallback.fit(interactions)

        user_idx = interactions["u"].to_numpy(dtype=int)
        item_idx = interactions["i"].to_numpy(dtype=int)
        ratings = interactions["rating"].to_numpy(dtype=float)

        n_users = int(user_idx.max()) + 1
        n_items = int(item_idx.max()) + 1
        matrix = csr_matrix((ratings, (user_idx, item_idx)), shape=(n_users, n_items))

        k = min(self.n_factors, min(matrix.shape) - 1)
        if k < 1:
            self.user_factors_ = None
            self.item_factors_ = None
            self.singular_values_ = None
            self.user_index_ = {
                int(user_id): int(u_idx)
                for user_id, u_idx in interactions[["user_id", "u"]].drop_duplicates().to_numpy()
            }
            self.recipe_index_ = {
                int(recipe_id): int(i_idx)
                for recipe_id, i_idx in interactions[["recipe_id", "i"]].drop_duplicates().to_numpy()
            }
            return self

        u, s, vt = svds(matrix, k=k)
        idx = np.argsort(-s)
        self.singular_values_ = s[idx]
        self.user_factors_ = u[:, idx]
        self.item_factors_ = vt[idx, :].T

        self.user_ids_ = np.sort(interactions["user_id"].unique())
        self.recipe_ids_ = np.sort(interactions["recipe_id"].unique())
        self.user_index_ = {
            int(user_id): int(u_idx)
            for user_id, u_idx in interactions[["user_id", "u"]].drop_duplicates().to_numpy()
        }
        self.recipe_index_ = {
            int(recipe_id): int(i_idx)
            for recipe_id, i_idx in interactions[["recipe_id", "i"]].drop_duplicates().to_numpy()
        }
        return self

    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self.user_index_

    def _predict_known_user(self, user_id: int, recipe_ids: list[int]) -> dict[int, float]:
        if self.user_factors_ is None or self.item_factors_ is None or self.singular_values_ is None:
            raise RuntimeError("Model must be fit before prediction")

        user_idx = self.user_index_[int(user_id)]
        user_vec = self.user_factors_[user_idx] * self.singular_values_
        scores: dict[int, float] = {}
        fallback_ids: list[int] = []

        known_indices: list[int] = []
        known_recipe_ids: list[int] = []
        for recipe_id in recipe_ids:
            recipe_id = int(recipe_id)
            item_idx = self.recipe_index_.get(recipe_id)
            if item_idx is None:
                fallback_ids.append(recipe_id)
                continue
            known_recipe_ids.append(recipe_id)
            known_indices.append(item_idx)

        if known_indices:
            item_matrix = self.item_factors_[known_indices]
            preds = self.global_mean + item_matrix @ user_vec
            for recipe_id, pred in zip(known_recipe_ids, preds, strict=True):
                scores[recipe_id] = float(pred)

        if fallback_ids:
            scores.update(self._popularity_fallback.score(fallback_ids))

        return scores

    def score(self, user_id: int, recipe_ids: list[int]) -> dict[int, float]:
        if not recipe_ids:
            return {}
        if not self.has_user(user_id) or self.user_factors_ is None:
            return self._popularity_fallback.score(recipe_ids)
        return self._predict_known_user(user_id, recipe_ids)

    def rank(self, user_id: int, recipe_ids: list[int], top_n: int = 10) -> list[tuple[int, float]]:
        scored = self.score(user_id, recipe_ids)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_n]
