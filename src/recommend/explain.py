"""Human-readable recommendation explanations."""

from __future__ import annotations

from src.models.session_context import SessionContext


def build_explanation(
    *,
    source: str,
    pantry_score: float,
    time_score: float,
    intent_score: float,
    minutes: int | None,
    context: SessionContext,
) -> str:
    parts: list[str] = []

    if context.meal_intent and intent_score >= 1.0:
        parts.append(f"matches {context.meal_intent}")
    if context.max_minutes and minutes is not None:
        parts.append(f"{minutes} min")
    if context.pantry and pantry_score > 0:
        parts.append(f"{pantry_score:.0%} pantry match")
    if source == "matrix_factorization":
        parts.append("based on your ratings")
    else:
        parts.append("popular pick")

    return "; ".join(parts)
