"""Session-level context for Stage 3 personalized re-ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from config.intents import MEAL_INTENT_TAGS


@dataclass(slots=True)
class SessionContext:
    """Optional query context: pantry, time budget, meal intent."""

    pantry: list[str] = field(default_factory=list)
    max_minutes: int | None = None
    meal_intent: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "SessionContext":
        pantry = data.get("pantry") or []
        if not isinstance(pantry, list):
            raise TypeError("pantry must be a list")
        return cls(
            pantry=pantry,
            max_minutes=data.get("max_minutes"),
            meal_intent=data.get("meal_intent"),
        )

    def to_dict(self) -> dict:
        return {
            "pantry": list(self.pantry),
            "max_minutes": self.max_minutes,
            "meal_intent": self.meal_intent,
        }

    def resolved_intent_tag(self) -> str | None:
        if not self.meal_intent:
            return None
        key = self.meal_intent.strip().lower()
        return MEAL_INTENT_TAGS.get(key)
