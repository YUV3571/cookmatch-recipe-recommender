#!/usr/bin/env python3
"""Step 5: verify unified profile filter API."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_recipes
from src.filter.profile_filter import (
    filter_recipes_by_profile,
    get_profile_violations,
    get_safe_recipe_ids,
    is_profile_compatible,
    summarize_profile_filter,
)
from src.models.user_profile import UserProfile


def main() -> int:
    print("=== Step 5: profile filter check ===")

    profiles = [
        UserProfile(diet="vegan", allergens=["nuts", "dairy", "gluten"]),
        UserProfile(diet="vegetarian", allergens=["dairy"]),
        UserProfile(diet=None, allergens=["nuts"]),
    ]

    print("\n[1/4] Demo user profiles")
    for profile in profiles:
        print(f"  {profile.to_dict()}")

    print("\n[2/4] Single-recipe checks")
    recipe_cases = [
        ("peanut butter cookies", ["peanut butter", "flour", "butter", "sugar"], profiles[0], False),
        ("veggie rice bowl", ["rice", "black beans", "tomatoes", "olive oil"], profiles[0], True),
        ("cheese pasta", ["pasta", "cheese", "tomato sauce"], profiles[1], False),
    ]
    for name, ingredients, profile, expected in recipe_cases:
        compatible = is_profile_compatible(ingredients, profile)
        violations = get_profile_violations(ingredients, profile)
        status = "OK" if compatible == expected else "FAIL"
        print(f"  [{status}] {name}: compatible={compatible}, violations={violations}")
        if compatible != expected:
            return 1

    print("\n[3/4] Funnel stats on sample (5000 recipes)")
    recipes = load_recipes(nrows=5000)
    for profile in profiles:
        summary = summarize_profile_filter(recipes, profile)
        print(f"  profile={profile.to_dict()}")
        for key, value in summary.items():
            print(f"    {key}: {value}")

    print("\n[4/4] Safe recipe IDs preview")
    safe_ids = get_safe_recipe_ids(recipes, profiles[0])[:5]
    safe = filter_recipes_by_profile(recipes, profiles[0]).head(5)
    print(f"  first safe ids: {safe_ids}")
    print("  first safe recipes:")
    for _, row in safe.iterrows():
        print(f"    - {row['id']}: {row['name']}")

    print("\nStage 1 complete. Next: Stage 2 (CF + popularity baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
