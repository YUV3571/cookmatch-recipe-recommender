# Graph Report - .  (2026-07-30)

## Corpus Check
- Corpus is ~9,414 words - fits in a single context window. You may not need a graph.

## Summary
- 301 nodes · 772 edges · 11 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Profile Filter API
- Ingredient Parsing
- Matrix Factorization
- Stage 3 Re-ranking
- Data Loading
- Offline Evaluation
- Allergen Filtering
- Colab Bootstrap
- Diet Filtering

## God Nodes (most connected - your core abstractions)
1. `UserProfile` - 48 edges
2. `Stage2Recommender` - 26 edges
3. `Stage3Recommender` - 24 edges
4. `load_recipes()` - 23 edges
5. `SessionContext` - 21 edges
6. `MatrixFactorizationModel` - 18 edges
7. `run_ablation()` - 17 edges
8. `PopularityModel` - 17 edges
9. `is_diet_compatible()` - 16 edges
10. `keyword_in_ingredients()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `load_recipes()`  [EXTRACTED]
  scripts/check_allergen_filter.py → src/data/loader.py
- `main()` --calls--> `load_recipes()`  [EXTRACTED]
  scripts/check_diet_filter.py → src/data/loader.py
- `main()` --calls--> `run_ablation()`  [EXTRACTED]
  scripts/check_evaluation.py → src/eval/offline_eval.py
- `main()` --calls--> `load_recipes()`  [EXTRACTED]
  scripts/check_ingredient_parser.py → src/data/loader.py
- `main()` --calls--> `load_recipes()`  [EXTRACTED]
  scripts/check_profile_filter.py → src/data/loader.py

## Import Cycles
- None detected.

## Communities (11 total, 0 thin omitted)

### Community 0 - "Profile Filter API"
Cohesion: 0.11
Nodes (32): main(), main(), _recipe_limit(), filter_recipes_by_profile(), get_profile_violations(), get_safe_recipe_ids(), is_profile_compatible(), DataFrame (+24 more)

### Community 1 - "Ingredient Parsing"
Cohesion: 0.09
Nodes (38): Meal intent tag mappings for Stage 3 content filtering., main(), ingredient_tokens(), ingredients_to_text(), keyword_in_ingredients(), normalize_text(), parse_ingredient(), parse_ingredients() (+30 more)

### Community 2 - "Matrix Factorization"
Cohesion: 0.10
Nodes (16): Project-wide settings for the recipe recommender., MatrixFactorizationModel, DataFrame, Matrix factorization recommender (truncated SVD) for Stage 2., User-item MF model trained with scipy.sparse.linalg.svds., PopularityModel, DataFrame, Popularity baseline for cold-start and fallback scoring. (+8 more)

### Community 3 - "Stage 3 Re-ranking"
Cohesion: 0.12
Nodes (17): Default Stage 3 re-ranking weights., main(), _recipe_limit(), Session-level context for Stage 3 personalized re-ranking., Optional query context: pantry, time budget, meal intent., SessionContext, build_explanation(), Human-readable recommendation explanations. (+9 more)

### Community 4 - "Data Loading"
Cohesion: 0.13
Nodes (26): Diet and allergen keyword maps for Stage 1 filtering.  These lists will drive ha, main(), _recipe_limit(), main(), build_eval_recipe_catalog(), eval_catalog_coverage(), filter_interactions_to_catalog(), get_dataset_path() (+18 more)

### Community 5 - "Offline Evaluation"
Cohesion: 0.18
Nodes (25): average_metric(), _dedupe_preserve_order(), hit_rate_at_k(), ndcg_at_k(), precision_at_k(), Offline ranking metrics for recommender evaluation., recall_at_k(), _aggregate_user_metrics() (+17 more)

### Community 6 - "Allergen Filtering"
Cohesion: 0.16
Nodes (24): main(), filter_recipes_by_allergens(), find_allergen_hits(), get_allergen_violations(), is_allergen_safe(), DataFrame, Hard allergen checks for Stage 1 filtering., Return allergen keywords found in a recipe ingredient list. (+16 more)

### Community 7 - "Colab Bootstrap"
Cohesion: 0.13
Nodes (26): bind(), bootstrap_repo(), _clone_with_git(), _clone_with_zip(), _download(), download_repo(), load_bootstrap_module(), purge_cached_modules() (+18 more)

### Community 8 - "Diet Filtering"
Cohesion: 0.18
Nodes (22): main(), filter_recipes_by_diet(), find_diet_hits(), get_diet_violations(), is_diet_compatible(), DataFrame, Hard diet checks for Stage 1 filtering., Return blocked keywords found for the given diet. (+14 more)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UserProfile` connect `Profile Filter API` to `Stage 3 Re-ranking`, `Offline Evaluation`, `Allergen Filtering`?**
  _High betweenness centrality (0.219) - this node is a cross-community bridge._
- **Why does `Stage3Recommender` connect `Stage 3 Re-ranking` to `Profile Filter API`, `Offline Evaluation`, `Colab Bootstrap`?**
  _High betweenness centrality (0.205) - this node is a cross-community bridge._
- **Why does `Stage2Recommender` connect `Profile Filter API` to `Matrix Factorization`, `Stage 3 Re-ranking`, `Offline Evaluation`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `UserProfile` (e.g. with `EvalScenario` and `Recommendation`) actually correct?**
  _`UserProfile` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Stage2Recommender` (e.g. with `EvalScenario` and `UserProfile`) actually correct?**
  _`Stage2Recommender` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Stage3Recommender` (e.g. with `EvalScenario` and `SessionContext`) actually correct?**
  _`Stage3Recommender` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `SessionContext` (e.g. with `EvalScenario` and `Stage3Recommendation`) actually correct?**
  _`SessionContext` has 3 INFERRED edges - model-reasoned connections that need verification._