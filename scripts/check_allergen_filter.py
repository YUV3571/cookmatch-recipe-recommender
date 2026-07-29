#!/usr/bin/env python3
"""Step 3: verify hard allergen rules on edge cases and dataset sample."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_recipes
from src.filter.allergen_filter import (
    get_allergen_violations,
    is_allergen_safe,
    summarize_allergen_filter,
)


def main() -> int:
    print("=== Step 3: allergen filter check ===")

    cases = [
        {
            "name": "peanut sauce noodles",
            "ingredients": ["peanut sauce", "rice noodles", "lime"],
            "allergens": ["nuts"],
            "expected_safe": False,
        },
        {
            "name": "ghee rice",
            "ingredients": ["ghee", "basmati rice", "cardamom"],
            "allergens": ["dairy"],
            "expected_safe": False,
        },
        {
            "name": "flour tortillas",
            "ingredients": ["flour", "water", "salt", "oil"],
            "allergens": ["gluten"],
            "expected_safe": False,
        },
        {
            "name": "nutmeg cookies should not trip nuts",
            "ingredients": ["nutmeg", "flour", "sugar", "butter"],
            "allergens": ["nuts"],
            "expected_safe": True,
        },
        {
            "name": "safe rice bowl",
            "ingredients": ["rice", "black beans", "tomatoes", "cumin"],
            "allergens": ["nuts", "dairy", "gluten"],
            "expected_safe": True,
        },
    ]

    print("\n[1/3] Hand-picked allergen cases")
    for case in cases:
        violations = get_allergen_violations(case["ingredients"], case["allergens"])
        safe = is_allergen_safe(case["ingredients"], case["allergens"])
        ok = safe == case["expected_safe"]
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {case['name']}: safe={safe}, violations={violations}")
        if not ok:
            return 1

    print("\n[2/3] Combined allergen profile funnel (sample 5000 recipes)")
    recipes = load_recipes(nrows=5000)
    profile = ["nuts", "dairy", "gluten"]
    summary = summarize_allergen_filter(recipes, profile)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n[3/3] Spot-check blocked and safe examples from sample")
    blocked = []
    safe_examples = []
    for _, row in recipes.iterrows():
        violations = get_allergen_violations(row["ingredients"], profile)
        if violations and len(blocked) < 3:
            blocked.append((row["name"], violations))
        if not violations and len(safe_examples) < 3:
            safe_examples.append(row["name"])
        if len(blocked) >= 3 and len(safe_examples) >= 3:
            break

    print("  blocked examples:")
    for name, violations in blocked:
        print(f"    - {name}: {violations}")

    print("  safe examples:")
    for name in safe_examples:
        print(f"    - {name}")

    print("\nStep 3 OK. Ready for Step 4 (diet rules: vegetarian/vegan).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
