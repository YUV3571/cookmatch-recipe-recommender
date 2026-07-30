"""Offline evaluation and ablation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.intents import MEAL_INTENT_TAGS
from src.eval.metrics import (
    average_metric,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.filter.profile_filter import is_profile_compatible
from src.models.session_context import SessionContext
from src.models.user_profile import UserProfile
from src.recommend.stage2 import Stage2Recommender
from src.recommend.stage3 import Stage3Recommender

TAG_TO_INTENT = {tag: intent for intent, tag in MEAL_INTENT_TAGS.items()}


@dataclass(slots=True)
class EvalScenario:
    name: str
    profile: UserProfile
    context: SessionContext
    use_stage3: bool = True
    force_popularity: bool = False


def intent_from_recipe_tags(tags: list[str]) -> str | None:
    for tag in tags:
        normalized = str(tag).strip().lower()
        if normalized in TAG_TO_INTENT:
            return TAG_TO_INTENT[normalized]
    return None


def build_oracle_context(recipe_meta: dict, mode: str) -> SessionContext:
    if not recipe_meta:
        return SessionContext()

    if mode == "pantry":
        pantry = list(recipe_meta.get("ingredients") or [])[:5]
        return SessionContext(pantry=pantry)
    if mode == "time":
        minutes = recipe_meta.get("minutes")
        if minutes is None:
            return SessionContext()
        return SessionContext(max_minutes=int(minutes))
    if mode == "intent":
        intent = intent_from_recipe_tags(recipe_meta.get("tags") or [])
        return SessionContext(meal_intent=intent)
    if mode == "full":
        intent = intent_from_recipe_tags(recipe_meta.get("tags") or [])
        minutes = recipe_meta.get("minutes")
        return SessionContext(
            pantry=list(recipe_meta.get("ingredients") or [])[:5],
            max_minutes=int(minutes) if minutes is not None else None,
            meal_intent=intent,
        )
    return SessionContext()


def _relevant_items(
    held_out: pd.DataFrame,
    min_rating: float,
) -> dict[int, set[int]]:
    relevant: dict[int, set[int]] = {}
    for _, row in held_out.iterrows():
        if float(row["rating"]) < min_rating:
            continue
        user_id = int(row["user_id"])
        recipe_id = int(row["recipe_id"])
        relevant.setdefault(user_id, set()).add(recipe_id)
    return relevant


def _recommend_ids(
    *,
    stage2: Stage2Recommender,
    stage3: Stage3Recommender,
    scenario: EvalScenario,
    user_id: int,
    top_k: int,
) -> list[int]:
    if scenario.use_stage3:
        recs = stage3.recommend(scenario.profile, scenario.context, user_id=user_id, top_n=top_k)
        return [rec.recipe_id for rec in recs]

    recs = stage2.recommend(scenario.profile, user_id=user_id, top_n=top_k)
    return [rec.recipe_id for rec in recs]


def constraint_violation_rate(
    recommended_ids: list[int],
    profile: UserProfile,
    recipe_meta: dict[int, dict],
) -> float:
    if not recommended_ids:
        return 0.0
    violations = 0
    for recipe_id in recommended_ids:
        meta = recipe_meta.get(recipe_id)
        if meta is None:
            continue
        if not is_profile_compatible(meta["ingredients"], profile):
            violations += 1
    return violations / len(recommended_ids)


def _aggregate_user_metrics(
    metric_rows: list[dict[str, float]],
    k: int,
) -> dict[str, float | str]:
    return {
        "users_evaluated": float(len(metric_rows)),
        f"precision@{k}": round(average_metric([row["precision"] for row in metric_rows]), 4),
        f"recall@{k}": round(average_metric([row["recall"] for row in metric_rows]), 4),
        f"hit_rate@{k}": round(average_metric([row["hit"] for row in metric_rows]), 4),
        f"ndcg@{k}": round(average_metric([row["ndcg"] for row in metric_rows]), 4),
        "constraint_violation_rate": round(
            average_metric([row["violation"] for row in metric_rows]), 4
        ),
    }


def evaluate_scenario(
    *,
    stage2: Stage2Recommender,
    stage3: Stage3Recommender,
    scenario: EvalScenario,
    recipe_meta: dict[int, dict],
    user_ids: list[int],
    relevant_by_user: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float | str]:
    metric_rows: list[dict[str, float]] = []

    for idx, user_id in enumerate(user_ids):
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue

        effective_user = 900_000_000 + idx if scenario.force_popularity else user_id
        recommended = _recommend_ids(
            stage2=stage2,
            stage3=stage3,
            scenario=scenario,
            user_id=effective_user,
            top_k=k,
        )
        metric_rows.append(
            {
                "precision": precision_at_k(recommended, relevant, k),
                "recall": recall_at_k(recommended, relevant, k),
                "hit": hit_rate_at_k(recommended, relevant, k),
                "ndcg": ndcg_at_k(recommended, relevant, k),
                "violation": constraint_violation_rate(recommended, scenario.profile, recipe_meta),
            }
        )

    metrics = _aggregate_user_metrics(metric_rows, k)
    metrics["scenario"] = scenario.name
    return metrics


def evaluate_oracle_stage3(
    *,
    stage3: Stage3Recommender,
    profile: UserProfile,
    held_out: pd.DataFrame,
    recipe_meta: dict[int, dict],
    user_ids: list[int],
    relevant_by_user: dict[int, set[int]],
    context_mode: str,
    k: int = 10,
) -> dict[str, float | str]:
    held_out_by_user = held_out.set_index("user_id")
    metric_rows: list[dict[str, float]] = []

    for user_id in user_ids:
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue

        row = held_out_by_user.loc[user_id]
        target_recipe_id = int(row["recipe_id"])
        meta = recipe_meta.get(target_recipe_id, {})
        context = build_oracle_context(meta, context_mode)
        recs = stage3.recommend(profile, context, user_id=user_id, top_n=k)
        recommended = [rec.recipe_id for rec in recs]

        metric_rows.append(
            {
                "precision": precision_at_k(recommended, relevant, k),
                "recall": recall_at_k(recommended, relevant, k),
                "hit": hit_rate_at_k(recommended, relevant, k),
                "ndcg": ndcg_at_k(recommended, relevant, k),
                "violation": constraint_violation_rate(recommended, profile, recipe_meta),
            }
        )

    metrics = _aggregate_user_metrics(metric_rows, k)
    metrics["scenario"] = f"stage3_oracle_{context_mode}"
    return metrics


def run_ablation(
    *,
    recipes: pd.DataFrame,
    train_interactions: pd.DataFrame,
    held_out: pd.DataFrame,
    user_sample_size: int = 200,
    k: int = 10,
    min_rating: float = 4.0,
    random_seed: int = 42,
    stage2: Stage2Recommender | None = None,
    stage3: Stage3Recommender | None = None,
) -> pd.DataFrame:
    if stage3 is not None:
        stage2 = stage3.stage2
    elif stage2 is None:
        stage2 = Stage2Recommender().fit(recipes, train_interactions)

    if stage3 is None:
        stage3 = Stage3Recommender().fit(recipes, train_interactions)
    elif stage3.recipe_meta_:
        pass
    else:
        stage3.fit(recipes, train_interactions)

    recipe_meta = {
        int(row["id"]): {
            "ingredients": row["ingredients"],
            "minutes": int(row["minutes"]),
            "tags": row["tags"],
        }
        for _, row in recipes.iterrows()
    }

    relevant_by_user = _relevant_items(held_out, min_rating=min_rating)
    eligible_users = [uid for uid in relevant_by_user if stage2.cf_model.has_user(uid)]
    if not eligible_users:
        raise ValueError("No eligible users found for evaluation")

    sample_size = min(user_sample_size, len(eligible_users))
    sampled_users = (
        pd.Series(eligible_users).sample(n=sample_size, random_state=random_seed).astype(int).tolist()
    )

    open_profile = UserProfile(diet=None, allergens=[])
    strict_profile = UserProfile(diet="vegan", allergens=["nuts", "dairy", "gluten"])

    rows: list[dict[str, float | str]] = []

    base_scenarios = [
        EvalScenario("popularity_open", open_profile, SessionContext(), use_stage3=False, force_popularity=True),
        EvalScenario("stage2_mf_open", open_profile, SessionContext(), use_stage3=False),
        EvalScenario("stage3_strict_profile", strict_profile, SessionContext(), use_stage3=False),
    ]

    for scenario in base_scenarios:
        rows.append(
            evaluate_scenario(
                stage2=stage2,
                stage3=stage3,
                scenario=scenario,
                recipe_meta=recipe_meta,
                user_ids=sampled_users,
                relevant_by_user=relevant_by_user,
                k=k,
            )
        )

    for context_mode in ("pantry", "time", "intent", "full"):
        rows.append(
            evaluate_oracle_stage3(
                stage3=stage3,
                profile=open_profile,
                held_out=held_out,
                recipe_meta=recipe_meta,
                user_ids=sampled_users,
                relevant_by_user=relevant_by_user,
                context_mode=context_mode,
                k=k,
            )
        )

    return pd.DataFrame(rows)
