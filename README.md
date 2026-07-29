# CookMatch — Food Recipe Recommender

Safety-constrained hybrid recipe recommender for a university RS project.

## Pipeline

1. **Stage 1** — hard profile filter (diet + allergens)
2. **Stage 2** — matrix factorization (SVD) + popularity fallback
3. **Stage 3** — pantry / time / meal-intent re-ranking + explanations

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py
```

## Google Colab

Open [`notebooks/03_colab_full_run.ipynb`](notebooks/03_colab_full_run.ipynb) in Colab.

1. Clone this repo
2. Upload `kaggle.json` or set Kaggle env vars
3. Run all cells — dataset downloads via `kagglehub` (not committed to git)

## Kaggle dataset

- Slug: `shuyangli94/food-com-recipes-and-user-interactions`
- Downloaded at runtime via [kagglehub](https://github.com/Kaggle/kagglehub)
- CSV files are **not** stored in this repo (see `.gitignore`)

## Project layout

```
config/          settings, allergen rules, intent tags
src/filter/      Stage 1 parser + profile filter
src/recommend/   Stage 2–3 models
src/eval/        offline metrics + ablation
src/models/      UserProfile, SessionContext
notebooks/       EDA, eval, Colab full run
scripts/         step-by-step check scripts
tests/           unit tests
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FOOD_RS_DATA_PATH` | kagglehub cache | override dataset location |
| `STAGE2_RECIPE_LIMIT` | `5000` in scripts | recipe rows for quick runs; `0` = full catalog |
| `EVAL_USER_SAMPLE` | `100` | users in ablation eval |
| `EVAL_TOP_K` | `10` | metrics @ K |

## Tests

```bash
python -m pytest tests/ -q
```

## License

University coursework project.
