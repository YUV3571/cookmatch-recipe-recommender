from src.eval.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_and_recall_at_k() -> None:
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 9}
    assert precision_at_k(recommended, relevant, 5) == 0.4
    assert recall_at_k(recommended, relevant, 5) == 2 / 3


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k([10, 11, 12], {12, 99}, 3) == 1.0
    assert hit_rate_at_k([10, 11, 12], {99}, 3) == 0.0


def test_ndcg_at_k_perfect_ranking() -> None:
    recommended = [7, 8, 9]
    relevant = {7, 8}
    assert ndcg_at_k(recommended, relevant, 3) == 1.0
