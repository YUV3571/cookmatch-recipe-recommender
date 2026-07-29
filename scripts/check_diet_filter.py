#!/usr/bin/env python3
"""Step 4: verify diet rules on edge cases and dataset sample."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_recipes
from src.filter.diet_filter import (
    get_diet_violations,
    is_diet_compatible,
    summarize_diet_filter,
)


def main() -> int:
    print("=== Step 4: diet filter check ===")

    cases = [
        {
            "name": "chicken soup",
            "ingredients": ["chicken broth", "noodles", "carrots"],
            "diet": "vegetarian",
            "expected_compatible": False,
        },
        {
            "name": "cheese pizza",
            "ingredients": ["pizza crust", "tomato sauce", "mozzarella"],
            "diet": "vegetarian",
            "expected_compatible": True,
        },
        {
            "name": "cheese pizza vegan",
            "ingredients": ["pizza crust", "tomato sauce", "mozzarella"],
            "diet": "vegan",
            "expected_compatible": False,
        },
        {
            "name": "egg breakfast",
            "ingredients": ["eggs", "butter", "toast"],
            "diet": "vegan",
            "expected_compatible": False,
        },
        {
            "name": "plant bowl",
            "ingredients": ["rice", "black beans", "tomatoes", "olive oil"],
            "diet": "vegan",
            "expected_compatible": True,
        },
        {
            "name": "omnivore no restriction",
            "ingredients": ["chicken", "milk", "eggs"],
            "diet": None,
            "expected_compatible": True,
        },
    ]

    print("\n[1/3] Hand-picked diet cases")
    for case in cases:
        violations = get_diet_violations(case["ingredients"], case["diet"])
        compatible = is_diet_compatible(case["ingredients"], case["diet"])
        ok = compatible == case["expected_compatible"]
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {case['name']} ({case['diet']}): compatible={compatible}, violations={violations}")
        if not ok:
            return 1

    print("\n[2/3] Diet funnel on sample 5000 recipes")
    recipes = load_recipes(nrows=5000)
    for diet in ("vegetarian", "vegan"):
        summary = summarize_diet_filter(recipes, diet)
        print(f"  {diet}:")
        for key, value in summary.items():
            print(f"    {key}: {value}")

    print("\n[3/3] Spot-check examples")
    for diet in ("vegetarian", "vegan"):
        blocked = []
        compatible = []
        for _, row in recipes.iterrows():
            violations = get_diet_violations(row["ingredients"], diet)
            if violations and len(blocked) < 2:
                blocked.append((row["name"], violations))
            if not violations and len(compatible) < 2:
                compatible.append(row["name"])
            if len(blocked) >= 2 and len(compatible) >= 2:
                break

        print(f"  {diet} blocked:")
        for name, violations in blocked:
            print(f"    - {name}: {violations}")
        print(f"  {diet} compatible:")
        for name in compatible:
            print(f"    - {name}")

    print("\nStep 4 OK. Ready for Step 5 (profile filter API).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
