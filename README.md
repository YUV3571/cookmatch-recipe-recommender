# CookMatch — Food Recipe Recommender

Safety-constrained hybrid recipe recommender for a university RS project.

## Architecture graph

Built with [graphify](https://github.com/safishamsi/graphify) — open [`graphify-out/graph.html`](graphify-out/graph.html) in a browser for an interactive map (301 nodes, 11 communities). SVG for slides: [`graphify-out/graph.svg`](graphify-out/graph.svg).

Full project history, issues, and next steps: **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**.

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
streamlit run app.py
```

### Streamlit Web Demo

Launch the interactive web UI:

```bash
streamlit run app.py
```

## Google Colab + GitHub

Colab loads **only the notebook file** from GitHub — not the full repo. Always run the **Setup cell** first.

### Open in Colab (one click)

| Notebook | Link |
|----------|------|
| **Setup (run first)** | [00_colab_github_setup.ipynb](https://colab.research.google.com/github/YUV3571/cookmatch-recipe-recommender/blob/main/notebooks/00_colab_github_setup.ipynb) |
| **Full pipeline run** | [03_colab_full_run.ipynb](https://colab.research.google.com/github/YUV3571/cookmatch-recipe-recommender/blob/main/notebooks/03_colab_full_run.ipynb) |

### Steps

1. Open **00** or run **cell 1** of **03** — clones `YUV3571/cookmatch-recipe-recommender` into `/content/`
2. Set `KAGGLE_API_TOKEN` (Kaggle → Settings → API → Create New Token)
3. Run remaining cells — dataset downloads via `kagglehub` (not in git)

If `git clone` fails in Colab, setup auto-falls back to GitHub zip download.

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
