#!/usr/bin/env python3
"""Stage 2: verify CF + popularity recommender on profile-safe pool."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_interaction_split, load_recipes
from src.models.user_profile import UserProfile
from src.recommend.stage2 import Stage2Recommender


def _recipe_limit() -> int | None:
    raw = os.getenv("STAGE2_RECIPE_LIMIT", "5000")
    if raw.lower() in {"0", "all", "full", "none"}:
        return None
    return int(raw)


def main() -> int:
    print("=== Stage 2: CF + popularity check ===", flush=True)
    recipe_limit = _recipe_limit()

    print("\n[1/4] Load recipes + train interactions", flush=True)
    recipes = load_recipes(nrows=recipe_limit, columns=["id", "name", "ingredients"])
    train = load_interaction_split("train")
    print(f"  recipes loaded: {len(recipes)} (limit={recipe_limit or 'full catalog'})", flush=True)
    print(f"  train interactions: {len(train)}", flush=True)

    print("\n[2/4] Train Stage 2 recommender", flush=True)
    t0 = time.time()
    recommender = Stage2Recommender().fit(recipes, train)
    print(f"  training done in {time.time() - t0:.2f}s", flush=True)

    profiles = [
        UserProfile(diet="vegan", allergens=["nuts", "dairy", "gluten"]),
        UserProfile(diet="vegetarian", allergens=["dairy"]),
    ]
    known_user = int(train["user_id"].iloc[0])
    cold_user = 999_999_999

    print("\n[3/4] Stage 1 funnel for demo profiles", flush=True)
    for profile in profiles:
        summary = recommender.summarize_stage1_funnel(profile)
        print(
            f"  profile={profile.to_dict()} -> safe={summary['safe_recipes']} ({summary['safe_pct']}%)",
            flush=True,
        )

    print("\n[4/4] Top recommendations", flush=True)
    for label, user_id in [("known user (MF)", known_user), ("cold user (popularity)", cold_user)]:
        profile = profiles[0]
        recs = recommender.recommend(profile, user_id=user_id, top_n=5)
        print(f"  {label}, profile={profile.to_dict()}")
        for rec in recs:
            print(f"    - {rec.recipe_id}: {rec.score:.3f} [{rec.source}] {rec.name}")

    print("\nStage 2 OK. Next: Stage 3 (pantry/time/intent re-rank).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
