"""User profile schema for Stage 1 filtering."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UserProfile:
    """Hard-constraint profile used by Stage 1."""

    diet: str | None = None
    allergens: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        diet = data.get("diet")
        allergens = data.get("allergens") or []
        if not isinstance(allergens, list):
            raise TypeError("allergens must be a list")
        return cls(diet=diet, allergens=allergens)

    def to_dict(self) -> dict:
        return {
            "diet": self.diet,
            "allergens": list(self.allergens),
        }
