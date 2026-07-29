#!/usr/bin/env python3
"""Evaluation: offline metrics + ablation table."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_interaction_split, load_recipes
from src.eval.offline_eval import run_ablation


def _recipe_limit() -> int | None:
    raw = os.getenv("STAGE2_RECIPE_LIMIT", "5000")
    if raw.lower() in {"0", "all", "full", "none"}:
        return None
    return int(raw)


def main() -> int:
    print("=== Evaluation + ablation check ===", flush=True)
    recipe_limit = _recipe_limit()
    user_sample = int(os.getenv("EVAL_USER_SAMPLE", "100"))
    top_k = int(os.getenv("EVAL_TOP_K", "10"))

    print("\n[1/3] Load data", flush=True)
    recipes = load_recipes(nrows=recipe_limit, columns=["id", "name", "ingredients", "minutes", "tags"])
    train = load_interaction_split("train")
    validation = load_interaction_split("validation")
    print(f"  recipes: {len(recipes)}", flush=True)
    print(f"  train interactions: {len(train)}", flush=True)
    print(f"  validation rows: {len(validation)}", flush=True)

    print(f"\n[2/3] Run ablation (users={user_sample}, k={top_k})", flush=True)
    t0 = time.time()
    results = run_ablation(
        recipes=recipes,
        train_interactions=train,
        held_out=validation,
        user_sample_size=user_sample,
        k=top_k,
    )
    print(f"  completed in {time.time() - t0:.2f}s", flush=True)

    print("\n[3/3] Ablation table", flush=True)
    print(results.to_string(index=False), flush=True)

    strict = results[results["scenario"] == "stage3_strict_profile"]
    if not strict.empty and float(strict.iloc[0]["constraint_violation_rate"]) != 0.0:
        print("\nFAIL: strict profile should have 0 constraint violations", flush=True)
        return 1

    print("\nEvaluation OK. Use notebooks/02_evaluation_ablation.ipynb for report plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
