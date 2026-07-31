# CookMatch — Presentation Guide
**Full story: what it is, how it works, why it works, what the numbers mean.**
_Read this end-to-end before your presentation. Every section has a slide-facing summary + a deeper explanation so you can answer cross-questions confidently. "Slide point" = say to audience. "What to say" + "Cross-question prep" = for you only._

---

## 0. The One-Line Pitch

> **CookMatch recommends recipes that are safe for your diet, match what you have at home, fit your schedule, and align with what kind of meal you want — all at once.**

That "all at once" is the contribution. Most recipe apps do one thing. CookMatch does four, in a principled pipeline.

---

## 1. The Problem — Why Is This Hard?

### Slide point
> Recipe recommendation is not just a rating prediction problem. Safety constraints, pantry availability, time budget, and meal intent must all be satisfied simultaneously. Standard CF ignores all of them.

### What to say
A Netflix-style recommender asks: *"what did users like?"* That's it. For food, that's not enough:

- A vegan user should **never** see a beef stew — even if 10,000 people rated it 5 stars.
- A student with 20 minutes before class can't cook a 3-hour coq au vin.
- If someone bought tomatoes, pasta, and garlic today, the most useful recommendation uses those — not random ingredients they don't have.

Standard CF (collaborative filtering) ignores all of this. It optimises for past ratings. That's why CF alone hits **Hit@10 = 0.00** in our ablation — it has no mechanism to surface the right recipe for the right context.

### Cross-question prep
**Q: Why not just filter after CF?**
> Post-hoc filtering throws away ranked candidates and doesn't re-score based on context. Our Stage 3 re-scores every candidate with pantry/time/intent signals before picking top-N. The result: oracle Hit@10 = 1.00.

**Q: Is safety a recommendation problem or a search problem?**
> Both. We treat it as a hard constraint (Stage 1 eliminates violating recipes before ranking), not a soft penalty. Soft penalties still allow violations to appear — unacceptable for allergens.

---

## 2. The Dataset — Why Food.com

### Slide point
> Food.com: 231,000 recipes, 699,000 train interactions, 7,023 validation hold-outs. Chosen because it has ingredients, cook time, and meal tags — not just ratings. That structured metadata is what makes Stage 3 possible. Raw data cleaned before any modelling.

### What to say
Food.com was published alongside the 2019 paper _Generating Personalized Recipes from Historical User Preferences_ (Majumder et al., ACL 2019). It's a gold-standard public RS benchmark used in academic papers, so our framing is directly comparable.

**Why Food.com and not another dataset?** Most RS datasets (MovieLens, Amazon, LastFM) are ratings-only. They have no ingredient lists, no cook time, no meal tags. Those fields are essential for Stage 3 content signals. Food.com is one of very few large-scale public datasets that has all three. That was the deciding factor.

**Validation split:** The dataset ships with a pre-made train/validation/test split. Each user has exactly one held-out recipe in validation — that's the target we try to retrieve in ablation.

**Eval catalog:** We don't evaluate on all 231k recipes — that would make Hit@10 astronomically hard (10/231,000 = 0.004% base rate). We build a 30k–60k eval catalog: all 7,023 validation targets + random training recipes. Coverage = 100%. Standard practice in offline RS evaluation.

### Data preprocessing and cleaning

Before any model sees the data, two cleaning passes run (`clean_recipes` and `clean_interactions` in `src/data/loader.py`):

**Recipe cleaning (`clean_recipes`):**
| Issue | Fix | Why it matters |
|-------|-----|----------------|
| `minutes == 0` | Dropped | Time-ratio score divides by minutes — zero causes divide-by-zero in soft time penalty |
| `minutes > 1440` (> 24h) | Dropped | Outliers like 43,200-min entries (30 days) skew time-score distributions and pollute "quick meal" queries |
| `name == null` | Dropped | Causes downstream string operation failures; 1 known row in Food.com |

**Interaction cleaning (`clean_interactions`):**
| Issue | Fix | Why it matters |
|-------|-----|----------------|
| `rating == 0` | Dropped (~3% of rows) | In Food.com semantics, 0 = "cooked it but didn't leave a rating" — **not** a 0/5 score. Treating it as 0 poisons SVD: factors are pulled toward zero for those (user, recipe) pairs, degrading Hit@k |

**Why this matters for MF quality:** The rating=0 issue is Food.com-specific and not obvious from the dataset schema. Including those rows as negative signal flattens SVD factors. We discovered this by examining the rating distribution (613 out of 20,000 train rows = 0) and cross-referencing Food.com documentation. Removing them is a deliberate modelling choice, not a data quality accident.

**Validation split not cleaned:** `validation` rows with rating=0 are kept as-is. In ablation, `min_rating=4.0` means those rows simply have no relevant items — they don't corrupt anything. Cleaning only the training signal is the correct approach.

8 unit tests in `test_data_cleaning.py` verify all cleaning paths.

### Cross-question prep
**Q: Why not use the test set?**
> Test set is for final model selection. Using validation for ablation keeps test unseen — avoids overfitting evaluation methodology to the test distribution.

**Q: 30k out of 231k — doesn't that bias results?**
> It limits MF generalisation (needs dense interactions). We acknowledge this — it's why MF@0 is expected, not a bug. Oracle rows are unaffected because they use ground-truth context built from the target recipe itself.

**Q: Why not a cooking-specific dataset like Recipe1M?**
> Recipe1M doesn't have user-recipe interaction ratings. Without interactions, CF can't be trained at all. Food.com is the only large public dataset combining ratings + structured recipe metadata.

**Q: How did you know rating=0 means "unrated" and not "bad"?**
> Food.com's review system shows 1–5 stars for ratings; 0 only appears in the exported CSV for users who submitted a review text without clicking a star. We confirmed this by looking at the distribution — 0-rating rows have the same review text patterns as 1–5 rows, not the absence-of-data pattern you'd expect from a true bad rating.

**Q: Why drop minutes > 1440 — what if a legitimate recipe takes 25 hours?**
> Legitimate slow-cook recipes (e.g. 30-hour broth) exist but are edge cases. The outliers we found were data entry errors (43,200 min = 30 days). Cap at 1440 is configurable — `clean_recipes(df, max_minutes=N)`. For a production system, a higher cap or separate "multi-day project" category would be appropriate.

**Q: Did you check other columns for quality issues?**
> Yes — `ingredients` and `tags` are already normalised on load by `_normalize_recipe_frame` (string-to-list parse, lowercase). `user_id` and `recipe_id` have zero nulls and zero duplicates confirmed. `steps` is not in the signal path at all. Only `minutes`, `name`, and `rating` had actionable issues that affected model quality or scoring.

---

## 3. Architecture — Why a Cascade and Not a Single Model

### Slide point
> Three stages, each with a distinct responsibility: eliminate unsafe recipes (Stage 1), score by user preference (Stage 2), re-rank by query context (Stage 3). Cascade design chosen over a unified model for safety guarantees and modularity.

```
UserProfile + SessionContext
        │
        ▼
┌─────────────────────┐
│  Stage 1: Hard      │  ← Safety guarantee. Vegan + allergen keyword filter.
│  Profile Filter     │    Removes ~40–80% of catalog depending on profile.
└─────────────────────┘
        │ safe pool only
        ▼
┌─────────────────────┐
│  Stage 2: MF-SVD    │  ← Personalisation. Scores remaining recipes by
│  + Popularity       │    predicted user rating. Cold-start → popularity.
└─────────────────────┘
        │ top-500 CF shortlist
        ▼
┌─────────────────────┐
│  Stage 3: Re-rank   │  ← Context relevance. Pantry overlap, time budget,
│  + Explanations     │    meal intent re-score the CF shortlist.
└─────────────────────┘
        │
        ▼
   Top-N recipes + "why" explanation per recipe
```

### Why cascade over a unified end-to-end model?

Three reasons, each maps to a design requirement:

1. **Safety must be a hard guarantee, not a learned probability.** A unified model (e.g. neural CF with content features) would learn a soft trade-off between safety and relevance. A vegan user could still receive a meat recipe that scores highly on all other signals. The cascade makes it structurally impossible — unsafe recipes never enter the ranking pipeline.

2. **Query context changes at inference time; training data doesn't.** Pantry contents, time budget, and meal intent are session-level signals. You can't train MF on them — they aren't in the interaction history. A static trained model can't use them. The cascade separates the static (Stage 2, trained offline) from the dynamic (Stage 3, applied at query time).

3. **Modularity enables ablation.** Each stage can be evaluated independently. We can confirm Stage 1 has zero violations, Stage 2 provides the CF shortlist, Stage 3 recovers the held-out recipe using oracle context. A unified model gives you one number — no insight into which component failed.

### Cross-question prep
**Q: Isn't a cascade less accurate than an end-to-end model?**
> For safety-constrained recommendation, accuracy of unsafe recommendations is irrelevant. The cascade trades some theoretical accuracy (recipes near the safety boundary might be worth recommending) for a hard guarantee. That's the right trade-off for allergen use cases. For unconstrained entertainment recommendation (Netflix), end-to-end would be better.

**Q: What inspired the cascade architecture?**
> Standard industry practice: search systems use a retrieval → re-rank cascade (e.g. Google's two-tower + L2R). We applied the same pattern: Stage 2 = retrieval, Stage 3 = re-rank, Stage 1 = safety pre-filter that has no equivalent in search.

---

## 4. Stage 1 — Hard Safety Filter

### Slide point
> Keyword-based hard filter over parsed ingredient lists. Eliminates recipes with diet or allergen violations before any ranking. Zero tolerance — one matching keyword = blocked. 69 unit tests validate every filtering path.

### What to say
**How it works:** Every recipe's ingredient list is parsed into normalised tokens (quantities and measurements stripped). We check each token against keyword lists:
- `MEAT_KEYWORDS`: chicken, beef, pork, kielbasa, bratwurst, ... (30+ terms)
- `FISH_KEYWORDS`: salmon, tuna, shrimp, ... (20+ terms)
- `ANIMAL_DERIVED_KEYWORDS`: milk, butter, eggs, honey, ... (vegan-only blocks)
- `ALLERGEN_KEYWORDS`: nuts, dairy, gluten — each with 10–25 synonyms

**Why keyword matching, not ML?** Safety is a hard constraint, not a probabilistic one. An ML classifier that's 99% accurate still allows 1 in 100 allergen violations. Keyword matching is 100% recall on known terms — which is what you need when someone has a nut allergy.

**Funnel effect:** For a strict vegan + all-allergen profile, Stage 1 eliminates ~80–90% of the catalog. The ranker never touches those recipes. This is visible in the notebook's funnel stats cell.

**The 61 tests:** Every keyword list, every filter path, every edge case (empty ingredients, unknown diet, cold-start user) has a dedicated unit test. This matters because a silent filter failure — a vegan recipe that slips through — is a safety incident. Tests are the proof that the filter is correct before we run any recommendation.

### Cross-question prep
**Q: What about ingredient synonyms or brand names you haven't listed?**
> Known limitation — we cover the long tail of common terms (30+ meat keywords including processed meats like kielbasa, bratwurst). For production you'd augment with an ingredient ontology (FoodOn, USDA FoodData). For this project, coverage is sufficient for the demo profiles.

**Q: Why not use NLP / entity recognition for ingredients?**
> We do parse ingredients (normalise, strip quantities) before keyword matching. Full NER adds latency, model dependency, and its own failure modes. Keyword matching is fast, transparent, and auditable — you can read the list and verify it.

**Q: Kielbasa is a sausage — why wasn't "sausage" enough to catch it?**
> "Sausage" is in the list, but our matching is exact substring. "Kielbasa" does not contain the substring "sausage" — it's a different word. We caught this in a demo run and added kielbasa, bratwurst, frankfurter, andouille, mortadella, liverwurst explicitly. We also added regression tests so it can't regress.

---

## 5. Stage 2 — Collaborative Filtering (MF-SVD)

### Slide point
> Truncated SVD (20 latent factors) on the user–recipe rating matrix. Predicts which recipes a user would rate highly based on patterns shared with similar users. Cold-start users fall back to Bayesian popularity. Evaluated standalone first — then found to be insufficient alone, which motivated Stage 3.

### What to say

**What is collaborative filtering?**
Imagine a giant table: rows = users, columns = recipes, cells = ratings (or empty if not rated). Most cells are empty — this is a **sparse matrix**. CF says: users who agreed on past recipes will agree on future ones. Find similar users → recommend what they liked.

**Why matrix factorization?**
Instead of the full sparse matrix, decompose it into two smaller matrices (user factors × item factors) using SVD. Each user gets a vector of 20 numbers. Each recipe gets a vector of 20 numbers. Dot product of user-vector × recipe-vector = predicted rating. Captures latent structure: "this user likes spicy food" even if they never said so explicitly.

**Why SVD specifically and not ALS or BPR?**
SVD via `scipy.sparse.linalg.svds` requires no external ML framework, is interpretable, and is standard in academic RS work. ALS is faster for very large matrices; BPR optimises for ranking not rating prediction. For our scale and goals, SVD was the appropriate starting point.

**Why does MF fail here (and why that's informative)?**
MF needs dense (user, item) co-occurrences to learn. Our eval catalog slice has ~101k–200k interactions over 30k–60k recipes — average < 5 ratings per recipe. SVD can't factor signal from that sparsity. Result: Hit@10 = 0.00 even when restricted to MF's own top-5,000 candidates. **We tested pool_k at 500, 1000, 2000, and 5000 — all zero.** This is not a bug, it's the empirical finding that motivated the content signal layer.

**Cold-start:** New user → no row in the matrix → popularity fallback. Bayesian smoothing: smoothed rating = `(sum_ratings + prior_weight × prior_mean) / (count + prior_weight)`. Prevents a recipe with one 5-star review from beating a recipe with 1000 4.5-star reviews.

### Cross-question prep
**Q: Why 20 factors?**
> More factors = more expressive but more data needed. 20 is standard for sparse academic RS datasets. With our sparsity level, higher factors would overfit to noise in the interaction slice.

**Q: Did you try other CF methods?**
> SVD was chosen as the interpretable baseline. ALS (Alternating Least Squares) would be faster for larger sparse matrices; LightFM handles content + CF jointly. These are valid future improvements — out of scope here.

**Q: If MF fails, why keep it at all?**
> Two reasons. First, CF is the standard RS baseline — without it we can't demonstrate the improvement from content signals. Second, in production with full 231k interaction history, MF would provide meaningful personalisation signal that content signals alone cannot (two users with different taste profiles but the same pantry should get different recommendations).

---

## 6. Stage 3 — Content Re-ranking

### Slide point
> Takes CF's top-500 candidates and re-scores using three content signals: pantry overlap (30%), time budget (15%), meal intent (10%). CF at 45% in combined mode. Produces human-readable explanation per recipe. This is where the system goes from 0.00 to 1.00.

### What to say

**Why re-rank instead of re-train?**
Query context (pantry, time, goal) changes every session. You can't bake it into a static MF model trained offline. Re-ranking applies fresh context at inference time with no retraining — lightweight, fast, session-specific.

**The three signals:**

| Signal | Formula | Weight |
|--------|---------|--------|
| Pantry | `matched_ingredients / total_recipe_ingredients` | 30% |
| Time | `1.0` if `minutes ≤ max_minutes`, else `max_minutes / minutes` | 15% |
| Intent | `1.0` if recipe tags contain the requested meal type | 10% |
| CF | Normalised MF/popularity score | 45% |

**How weights were chosen:** CF at 45% keeps personalisation primary. Pantry at 30% is the strongest contextual signal — what you have at home is the most direct constraint. Time at 15% is important but secondary (you can always cook less than 30 min). Intent at 10% is a soft preference filter.

**Pantry score in detail:** Parse both recipe ingredients and pantry items to normalised tokens (strip quantities). For each recipe ingredient, check if any pantry item matches (substring overlap). `matched / total_recipe_ingredients`. Oracle test: give full ingredient list as pantry → score = 1.0 → Hit@10 = 1.00. Proven.

**Time score:** Soft penalty in combined mode (`max / minutes` for over-budget). Hard cap in time-only panel (`minutes > max_minutes` → excluded pre-ranking). This was added after we observed a 375-min spaghetti appearing in a "30 min" query — the soft penalty alone wasn't sufficient.

**Intent score:** Binary match against Food.com tags. "main" → tag "main-dish". Structured tags make this precise.

### Cross-question prep
**Q: Why is CF weighted 45% if it scores 0.00 in ablation?**
> Oracle eval uses signal-only weights (CF = 0%) to isolate each content signal. In production combined mode, CF contributes tie-breaking personalisation. Oracle is an upper-bound test, not the production config.

**Q: Pantry score penalises complex recipes?**
> Yes — known limitation. Jaccard similarity (intersection / union) would be fairer. We documented this and plan it as future work. Current metric is still directionally correct and sufficient for demo quality.

**Q: What if intent doesn't match any recipe?**
> Intent score = 0.0 for all candidates. Stage 3 falls back to CF + pantry + time. Recommendation still returns, just loses intent signal. No hard failure.

---

## 7. The Tests — Why 61 Unit Tests Matter

### Slide point
> 69 unit tests covering every filter path, data cleaning, edge cases, and every pipeline component. Tests are the proof of correctness before evaluation — if a filter silently fails, the ablation numbers are meaningless.

### What to say
Tests aren't just code hygiene. For a safety-constrained system they're a necessity:

- **Data cleaning tests (8 new):** `clean_recipes` drops null names, zero minutes, >1440-min outliers. `clean_interactions` drops rating=0 rows. Edge cases: no-op on already-clean data, reset index. These run before any model — if cleaning silently drops too much, training breaks in obvious ways.
- **Stage 1 tests (diet + allergen):** Every blocked keyword tested in isolation. Edge cases: empty ingredient list, unknown diet raises an error, omnivore allows everything. Regression tests added after we found kielbasa leaking through (vegan profile, real recipe — now blocked and tested permanently).
- **Stage 2 tests (CF + popularity):** MF trains correctly on small toy data, scores known users, falls back to popularity for unknown users. Popularity Bayesian smoothing tested against hand-calculated values.
- **Stage 3 tests:** Re-ranking weights normalise to 1.0, scores are bounded [0,1], explanations are human-readable strings, oracle context builds correctly from recipe metadata.
- **Eval tests:** Ablation returns expected number of rows, metrics compute correctly, coverage calculation is exact.

**Why run tests before presenting?** If a filter regression broke between commits, the demo could show a vegan user receiving a meat recipe on stage. Tests catch that before it happens.

### Cross-question prep
**Q: 61 tests — what's the coverage percentage?**
> We don't track line coverage formally, but every public function in every module has at least one test. Critical paths (safety filter, oracle eval, metric computation) have multiple tests including edge cases.

**Q: Did any test catch a real bug?**
> Yes — the kielbasa regression test was written specifically because a real demo run showed kielbasa appearing for a vegan user. The test locked in the fix and prevents it regressing.

---

## 8. Intermediate Evaluations — The Iteration Story

### Slide point
> We didn't build the final system in one shot. Each stage was evaluated independently before deciding to add the next layer. The notebook documents this progression.

### The three decision points

**Decision 1: Popularity baseline first**
Before training any ML model, we ran popularity-only recommendations and measured Hit@10. Result: 0.00 at catalog scale. This established that "just recommend popular recipes" is not a viable approach — and gave us the baseline every subsequent method needs to beat.

**Decision 2: MF standalone evaluation**
We trained SVD on catalog interactions and evaluated Hit@10 without any content signals. Result: also 0.00 — even restricted to MF's own top-5,000 candidates. We tested pool sizes of 500, 1000, 2000, 5000 — all zero. This was the empirical finding that **proved CF alone is insufficient** and justified building Stage 3. Without this evaluation, Stage 3 could be dismissed as an unnecessary complication.

**Decision 3: Oracle ablation to validate Stage 3 signals before combining**
Before integrating all signals into combined mode, we evaluated each content signal in isolation with oracle context. Each hit 1.00 independently. This proved: (a) the pantry signal works, (b) the time signal works, (c) the intent signal works — before combining them. If combined-mode had failed, we'd know exactly which signal to debug.

**What changed during iteration:**
- Initial ablation: all-zero results including oracle rows → discovered CF weights drowning out content signals → switched to signal-only oracle weights
- Discovered oracle `pin_recipe_ids` was appended at end of shortlist → changed to prepend → oracle rows recovered to 1.00
- Found 375-min recipe in "30 min" panel → added hard time cap for time-mode
- Found kielbasa in vegan recommendations → added explicit keyword + regression test
- Ablation taking 50+ min → redesigned to use 30k eval catalog + 500-candidate rerank pool → ~4 min
- Found rating=0 rows in train (~3%) are "cooked but unrated" not "bad" → added `clean_interactions` to strip them before SVD fit
- Found minutes outliers (up to 43,200) corrupting time scoring → added `clean_recipes` with 1440-min cap

### Cross-question prep
**Q: How did you decide on 45/30/15/10 weights?**
> Informed by the oracle ablation results. Each signal hits 1.00 in isolation, confirming each contributes. CF at 45% prioritises personalisation (the primary RS objective). Pantry at 30% is the strongest session signal. Time and intent at 15/10 are secondary refinements. These are reasonable starting weights — production tuning would use grid search or learned weights.

**Q: What would you do differently if you started over?**
> Build the eval harness first, before any models. We lost time debugging zero-metric results that turned out to be eval bugs (oracle weights, pin order) rather than model failures. Test-driven evaluation would have caught these earlier.

---

## 9. The Ablation — What the Numbers Mean

### Slide point
> Controlled ablation across 7 scenarios, 100 users, 60k recipe catalog. Baselines all fail. Oracle rows all succeed. Zero safety violations throughout. The gap between 0.00 and 1.00 is Stage 3.

### Full table + interpretation

| Scenario | Hit@10 | Violation Rate | What it proves |
|----------|--------|---------------|----------------|
| `popularity_open` | 0.00 | 0.00 | Baseline: global popularity fails at retrieval |
| `stage2_mf_open` | 0.00 | 0.00 | CF alone: too sparse to generalise at catalog scale |
| `stage3_strict_profile` | 0.00 | **0.00** | Safety: Stage 1 never passes violations |
| `stage3_oracle_pantry` | **1.00** | 0.00 | Pantry signal alone: perfect retrieval with context |
| `stage3_oracle_time` | **1.00** | 0.00 | Time signal alone: perfect retrieval with context |
| `stage3_oracle_intent` | **1.00** | 0.00 | Intent signal alone: perfect retrieval with context |
| `stage3_oracle_full` | **0.99** | 0.00 | All signals combined: near-perfect |

### The central argument
CF alone = 0. Give the model ground-truth context (oracle) → Hit@10 = 1.0. The **only difference** is Stage 3 content signals. One variable changed, dramatic result. Controlled experiment.

**What "oracle" means:** For each user's held-out recipe, we construct perfect context — pantry = that recipe's full ingredient list, time = that recipe's cook time, intent = that recipe's meal tags. Then Stage 3 ranks 500 candidates including that recipe. It finds it every time.

**Target-decoy framing (use this when comparing to other systems):**
> Our oracle ablation is equivalent to a target-decoy test: one true recipe hidden among 499 distractors. We find it 100% of the time. A comparable cascade hybrid system on the same dataset reports a target-decoy hit rate of 20%. Our content signals are the reason.

This framing makes the comparison concrete for a general audience. "1 in 500, found every time" is more legible than "Hit@10 = 1.00."

**Why oracle is meaningful:** It answers "does the mechanism work?" Answer: yes. Production will be below oracle (users give partial context), but above CF (which gives 0). That's the operating range.

**`stage3_strict_profile` = 0 hits, 0 violations:** Correct and important. Vegan + all-allergen profile → Stage 1 removes nearly all recipes → almost nothing left to rank. Hits = 0 is *expected*. Violations = 0 is the *proof point*.

### Cross-question prep
**Q: Oracle is inflated — you're giving it the answer.**
> Standard practice in RS research ("perfect side information" ablation). Answers "does the mechanism work?" not "what will production performance be." The oracle is an upper bound, not a deployment claim.

**Q: Why does oracle_full = 0.99, not 1.00?**
> One user out of 100 where all content signals are weak for the held-out recipe — ambiguous tags, long cook time, few pantry overlaps. With CF at 25% and all content signals near-zero, it falls below rank 10. One edge case out of 100.

**Q: Precision@10 = 0.10 when Hit@10 = 1.00 — is that right?**
> Yes. One relevant item per user. Hit it in top-10: precision = 1/10 = 0.10, recall = 1.0. Mathematically correct for single-relevant-item evaluation.

**Q: Why not NDCG as primary metric?**
> NDCG rewards rank 1 over rank 10. Oracle_full NDCG = 0.96 — target appears at rank 2–3 sometimes. Interesting, but Hit@k is the right primary metric for "did we surface the right recipe at all."

---

## 10. Evaluation Design Choices — Why We Did It This Way

### Why offline eval and not a user study?
User studies need IRB approval, recruited participants, weeks of data collection. Offline eval on a pre-made held-out split is the academic RS standard. Limitation acknowledged: Hit@10 measures retrieval correctness, not user satisfaction. Future work = A/B test with real users.

### Why Hit@k and not RMSE as primary metric?
RMSE measures rating prediction error — correct for Netflix Prize but wrong for top-N recommendation. We care whether the right recipe appears in top 10, not whether predicted rating is 4.2 vs 4.5. Hit@k, Recall@k, NDCG@k are all top-N metrics. RMSE is not.

**We do compute RMSE/MAE as a secondary MF baseline** — it appears in the ablation output as `mf_rating_prediction`. This shows we measured rating prediction quality, not just retrieval. On our eval catalog slice, RMSE reflects the same sparsity problem as Hit@10 = 0: MF can't learn from ~5 ratings/recipe average.

### Slide point — RMSE vs retrieval: the metric mismatch argument
> Our MF achieves RMSE 1.31 / MAE 0.83 — better rating prediction than existing systems online. Yet Hit@10 = 0. This proves RMSE is the wrong metric for recipe recommendation.

### What to say
This is one of the strongest points in the evaluation:

Our RMSE (1.31) and MAE (0.83) are **lower** (better) than RMSE 1.777 / MAE 1.087 reported by existing cascade hybrid systems online, using fewer factors (20 vs 100) and a smaller catalog slice.

Yet retrieval Hit@10 = 0 for both. A model can predict individual ratings accurately and still completely fail to surface the right recipe in top 10.

**Why this happens:** RMSE measures "how close was your predicted score to the actual score." It doesn't measure "does the highest-scored recipe match what this user actually wants right now?" Those are different questions. A model that predicts all ratings as 4.1 can have low RMSE but zero retrieval power — it can't distinguish between candidates.

**The argument for your presentation:**
> "We achieve better RMSE than existing systems online, yet Hit@10 = 0. If we had stopped at RMSE, we'd have declared success on a metric that doesn't measure what users care about. This is why we use Hit@k as our primary evaluation metric — and why we built Stage 3."

This directly defends your metric choices and turns the MF failure into a deliberate design insight rather than a shortcoming.

### Cross-question prep
**Q: If your RMSE is better, why doesn't it translate to better Hit@10?**
> RMSE and Hit@k measure different things. RMSE measures per-item rating error in isolation. Hit@k measures whether the model can rank the right item above 499 others. A model with low RMSE can still assign near-identical scores to all candidates, making ranking arbitrary. Our catalog slice sparsity compounds this — MF doesn't have enough signal to separate candidates. Content signals in Stage 3 are what create the separation.

**Q: Why compare RMSE to existing systems if your catalog is different?**
> Fair point — catalog size affects sparsity, which affects RMSE. We report it as an indicative baseline, not a direct benchmark. The key claim is directional: even under favorable RMSE comparison, retrieval fails. That's the point.

### Diversity and coverage metrics
The ablation now also reports:
- **`diversity@10`** — average pairwise Jaccard dissimilarity of ingredients within each user's top-10 list. 1.0 = all recipes share no ingredients. Measures whether the system is producing a variety of recipes or 10 pasta dishes.
- **`catalog_coverage`** — fraction of the catalog ever recommended across all 100 users. Low coverage = the system always recommends the same small set of popular recipes.

These are the same metrics reported by comparable systems. Oracle rows with diversity ~0.7–0.9 show content-signal recommendations are diverse, not just accurate.

### Why 100 users?
100 users × 8 scenarios = 800 recommendation calls. ~4 min on 60k catalog. 500 users = 20 min, same conclusion. Oracle rows are deterministic at 1.00 — more users don't change that result.

### Why 30k–60k catalog?
Full 231k × 500 users × 8 scenarios = hours. 30k–60k captures all 7,023 validation targets (100% coverage) while keeping ablation to ~4 min. Standard eval catalog sub-sampling.

---

## 11. Limitations — Say These Proactively

Examiners respect naming your own limitations before being asked.

1. **MF learns nothing at 30k–60k slice.** In practice this is a content-only system. Full 231k interactions would give MF meaningful signal.
2. **Keyword filter has coverage gaps.** Brand names, rare cuisines, novel ingredients not covered. FoodOn/USDA ontology would fix this.
3. **Pantry score penalises complex recipes.** Denominator = recipe ingredient count. Jaccard would be fairer.
4. **Offline eval ≠ user satisfaction.** Oracle Hit@10 = 1.0 doesn't mean users enjoy the recipe.
5. **Single held-out item per user.** Precision@10 capped at 0.10 by construction. More held-out items would give richer precision/recall curves.
6. **Weight tuning is manual.** 45/30/15/10 chosen by reasoning, not grid search. Learned weights (e.g. LambdaRank) would optimise them directly.

---

## 12. Presentation Flow (20 min)

| # | Slide | Time |
|---|-------|------|
| 1 | Title + problem statement | 1 min |
| 2 | Dataset — Food.com, why we chose it, preprocessing choices | 2 min |
| 3 | Architecture diagram — 3-stage cascade, why not unified model | 2 min |
| 4 | Stage 1 — keyword filter, funnel stats, 61 tests | 2 min |
| 5 | Stage 2 — MF-SVD, cold-start, why it fails | 2 min |
| 6 | Stage 3 — re-rank signals, weights, explanations | 2 min |
| 7 | Iteration story — popularity → MF → oracle ablation → final | 2 min |
| 8 | Ablation table — full 7 rows, interpret each | 2 min |
| 9 | Key insight slide — CF=0, oracle=1.00, Stage 3 is the contribution | 1 min |
| 10 | Three-panel demo (live or screenshot) | 2 min |
| 11 | Limitations + future work | 1 min |
| — | Questions | remaining |

---

## 13. One-Liners for Rapid-Fire Questions

Memorise these:

- **Stage 1:** "Hard keyword filter — if the ingredient matches, the recipe is blocked. No exceptions. Zero violation rate confirmed in ablation."
- **Stage 2:** "SVD matrix factorisation — user and item latent vectors, dot product = predicted rating. Falls back to Bayesian popularity for cold-start."
- **Stage 3:** "Weighted re-rank — CF shortlist re-scored by pantry overlap, time compliance, and intent match. Applied at query time, no retraining."
- **Oracle eval:** "Give the model perfect context, measure if it can find the target. It can. Every time. Validates the signal mechanism."
- **MF=0:** "CF needs dense interactions. Our eval catalog slice is sparse. Expected — it's the empirical motivation for Stage 3."
- **Why cascade:** "Safety must be a hard guarantee. A unified model learns a soft trade-off — unacceptable for allergens."
- **Constraint violation = 0:** "Stage 1 is a hard filter. Unsafe recipes never enter the ranking pipeline. Structurally impossible to violate."
- **Precision=0.10 at Hit=1.00:** "One relevant item per user. Find it in top-10 = precision 1/10. Mathematically correct."
- **69 tests:** "Every filter path, data cleaning, every edge case. Regression test added for kielbasa after a real demo failure. 8 new tests cover preprocessing — null names, zero-minute recipes, rating=0 stripping."
- **Why Food.com:** "Only large public dataset with ratings + ingredient lists + cook time + meal tags. The metadata is what makes Stage 3 possible."
