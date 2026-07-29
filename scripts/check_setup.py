#!/usr/bin/env python3
"""Step 0: verify environment, dataset access, and basic recipe loading."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.constraints import SUPPORTED_ALLERGENS, SUPPORTED_DIETS
from src.data.loader import get_dataset_path, load_recipes


def main() -> int:
    print("=== Step 0: setup check ===")

    print("\n[1/4] Supported Stage 1 constraints")
    print(f"  diets: {', '.join(SUPPORTED_DIETS)}")
    print(f"  allergens: {', '.join(SUPPORTED_ALLERGENS)}")

    print("\n[2/4] Resolve dataset path (kagglehub)")
    dataset_path = get_dataset_path()
    print(f"  path: {dataset_path}")

    files = sorted(p.name for p in dataset_path.iterdir() if p.is_file())
    print(f"  files: {', '.join(files)}")

    print("\n[3/4] Load sample recipes")
    recipes = load_recipes(nrows=5, columns=["id", "name", "ingredients", "tags", "minutes"])
    print(f"  loaded rows: {len(recipes)}")
    print(recipes[["id", "name", "minutes"]].to_string(index=False))

    print("\n[4/4] Parsed ingredient preview")
    first = recipes.iloc[0]
    print(f"  recipe: {first['name']}")
    print(f"  ingredients ({len(first['ingredients'])}): {first['ingredients'][:6]}")
    print(f"  tags ({len(first['tags'])}): {first['tags'][:6]}")

    print("\nStep 0 OK. Ready for Step 1 (EDA slice).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
