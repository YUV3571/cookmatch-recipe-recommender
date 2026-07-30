"""Offline ranking metrics for recommender evaluation."""

from __future__ import annotations

import math


def _dedupe_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = _dedupe_preserve_order(recommended)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top_k = _dedupe_preserve_order(recommended)[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top_k = _dedupe_preserve_order(recommended)[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top_k = _dedupe_preserve_order(recommended)[:k]
    if not top_k or not relevant:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_metric(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def intra_list_diversity(recommended_ids: list[int], recipe_meta: dict[int, dict]) -> float:
    """Average pairwise Jaccard dissimilarity of ingredients within the recommended list.

    1.0 = all recipes share no ingredients (maximum diversity).
    0.0 = all recipes have identical ingredient sets.
    """
    ids = recommended_ids[:10]
    ingredient_sets = []
    for rid in ids:
        meta = recipe_meta.get(rid)
        if meta is None:
            continue
        ings = meta.get("ingredients") or []
        ingredient_sets.append({str(i).strip().lower() for i in ings if i})

    if len(ingredient_sets) < 2:
        return 0.0

    total, pairs = 0.0, 0
    for i in range(len(ingredient_sets)):
        for j in range(i + 1, len(ingredient_sets)):
            a, b = ingredient_sets[i], ingredient_sets[j]
            union = a | b
            if not union:
                continue
            jaccard_similarity = len(a & b) / len(union)
            total += 1.0 - jaccard_similarity
            pairs += 1

    return total / pairs if pairs else 0.0


def catalog_coverage(all_recommended_ids: list[list[int]], catalog_size: int) -> float:
    """Fraction of catalog ever recommended across all users."""
    if catalog_size <= 0:
        return 0.0
    unique = {rid for user_recs in all_recommended_ids for rid in user_recs}
    return len(unique) / catalog_size
