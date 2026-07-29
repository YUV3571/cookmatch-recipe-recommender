#!/usr/bin/env python3
"""Step 2: verify ingredient parser on hand-picked and dataset examples."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_recipes
from src.filter.ingredient_parser import (
    ingredients_to_text,
    keyword_in_ingredients,
    parse_ingredient,
    parse_ingredients,
)


def main() -> int:
    print("=== Step 2: ingredient parser check ===")

    hand_picked = [
        "2 cups milk",
        "unsalted butter",
        "1/2 tsp salt",
        "2% low-fat milk",
        "ground turkey",
        "reese's peanut butter cups",
        "lettuce cups",
        "nutmeg",
    ]

    print("\n[1/3] Hand-picked normalization")
    for raw in hand_picked:
        print(f"  {raw!r:35} -> {parse_ingredient(raw)!r}")

    print("\n[2/3] Keyword checks (false-positive guard)")
    checks = [
        (["nutmeg", "flour"], "nut", False),
        (["reese's peanut butter cups"], "peanut", True),
        (["lettuce cups"], "nut", False),
        (["cream cheese"], "cream cheese", True),
        (["2 cups milk"], "milk", True),
    ]
    for ingredients, keyword, expected in checks:
        got = keyword_in_ingredients(ingredients, keyword)
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] {ingredients} contains {keyword!r}? {got} (expected {expected})")
        if got != expected:
            return 1

    print("\n[3/3] Dataset sample")
    recipes = load_recipes(nrows=3)
    for _, row in recipes.iterrows():
        parsed = parse_ingredients(row["ingredients"])
        print(f"  recipe: {row['name']}")
        print(f"    raw ({len(row['ingredients'])}): {row['ingredients'][:4]}")
        print(f"    parsed ({len(parsed)}): {parsed[:4]}")
        print(f"    blob: {ingredients_to_text(row['ingredients'])[:90]}...")

    print("\nStep 2 OK. Ready for Step 3 (allergen rules).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
