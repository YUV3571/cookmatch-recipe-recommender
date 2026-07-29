#!/usr/bin/env python3
"""Stage 3: verify pantry/time/intent re-ranking."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_interaction_split, load_recipes
from src.models.session_context import SessionContext
from src.models.user_profile import UserProfile
from src.recommend.stage3 import Stage3Recommender


def _recipe_limit() -> int | None:
    raw = os.getenv("STAGE2_RECIPE_LIMIT", "5000")
    if raw.lower() in {"0", "all", "full", "none"}:
        return None
    return int(raw)


def main() -> int:
    print("=== Stage 3: personalized re-rank check ===", flush=True)
    recipe_limit = _recipe_limit()

    print("\n[1/3] Load data + train Stage 3", flush=True)
    recipes = load_recipes(nrows=recipe_limit, columns=["id", "name", "ingredients", "minutes", "tags"])
    train = load_interaction_split("train")
    t0 = time.time()
    recommender = Stage3Recommender().fit(recipes, train)
    print(f"  recipes loaded: {len(recipes)}", flush=True)
    print(f"  training done in {time.time() - t0:.2f}s", flush=True)

    profile = UserProfile(diet="vegan", allergens=["nuts", "dairy", "gluten"])
    known_user = int(train["user_id"].iloc[0])

    scenarios = [
        (
            "Pantry cook",
            SessionContext(pantry=["tomato", "pasta", "garlic"], max_minutes=45),
        ),
        (
            "20-minute dinner",
            SessionContext(max_minutes=20, meal_intent="main"),
        ),
        (
            "Dessert craving",
            SessionContext(meal_intent="dessert"),
        ),
    ]

    print("\n[2/3] Scenario recommendations", flush=True)
    for label, context in scenarios:
        recs = recommender.recommend(profile, context, user_id=known_user, top_n=3)
        print(f"  [{label}] context={context.to_dict()}", flush=True)
        for rec in recs:
            print(
                f"    - {rec.recipe_id}: final={rec.final_score:.3f} "
                f"(cf={rec.cf_score:.2f}, pantry={rec.pantry_score:.2f}, "
                f"time={rec.time_score:.2f}, intent={rec.intent_score:.2f}) "
                f"{rec.name}",
                flush=True,
            )
            print(f"      why: {rec.explanation}", flush=True)

    print("\n[3/3] Stage 1 funnel reminder", flush=True)
    summary = recommender.summarize_stage1_funnel(profile)
    print(f"  safe recipes for profile: {summary['safe_recipes']} ({summary['safe_pct']}%)", flush=True)

    print("\nStage 3 OK. Full CookMatch pipeline core is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
