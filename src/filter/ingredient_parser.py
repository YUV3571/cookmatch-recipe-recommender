"""Normalize ingredient strings for constraint matching and pantry overlap."""

from __future__ import annotations

import re
from typing import Iterable

WHITESPACE_RE = re.compile(r"\s+")

UNIT_PREFIX_RE = re.compile(
    r"^\s*(?:\d+(?:[\./]\d+)?(?:\s*-\s*\d+(?:[\./]\d+)?)?)\s*"
    r"(?:cups?|c|tbsp|tablespoons?|tsp|teaspoons?|oz|ounces?|fl\s*oz|"
    r"lb|lbs|pounds?|grams?|g|kg|ml|milliliters?|l|liters?|pinch(?:es)?|"
    r"cloves?|cans?|packages?|pkgs?|slices?|pieces?|sticks?|heads?|"
    r"bunch(?:es)?|sprigs?|medium|large|small|whole)\s+",
    re.IGNORECASE,
)

LEADING_NUMBER_RE = re.compile(r"^\s*\d+(?:[\./]\d+)?%?\s+")

DESCRIPTOR_PREFIXES = (
    "unsalted ",
    "salted ",
    "fresh ",
    "frozen ",
    "dried ",
    "chopped ",
    "diced ",
    "minced ",
    "ground ",
    "whole ",
    "low-fat ",
    "fat-free ",
    "reduced-fat ",
    "cooked ",
    "raw ",
    "peeled ",
    "sliced ",
    "grated ",
    "shredded ",
    "boneless ",
    "skinless ",
)


def normalize_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value.strip().lower())


def parse_ingredient(raw: str) -> str:
    """Normalize one ingredient line to canonical lowercase text."""
    text = normalize_text(raw)

    while True:
        updated = UNIT_PREFIX_RE.sub("", text, count=1)
        if updated == text:
            break
        text = normalize_text(updated)

    text = LEADING_NUMBER_RE.sub("", text)
    text = normalize_text(text)

    changed = True
    while changed:
        changed = False
        for prefix in DESCRIPTOR_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                changed = True
                break
        text = normalize_text(text)

    return text


def parse_ingredients(raw_ingredients: Iterable[str]) -> list[str]:
    """Normalize and deduplicate ingredient strings while preserving order."""
    seen: set[str] = set()
    parsed: list[str] = []

    for raw in raw_ingredients:
        if raw is None:
            continue
        normalized = parse_ingredient(str(raw))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parsed.append(normalized)

    return parsed


def ingredients_to_text(ingredients: Iterable[str]) -> str:
    """Join normalized ingredients into one searchable blob."""
    return " | ".join(parse_ingredients(ingredients))


def ingredient_tokens(ingredients: Iterable[str]) -> set[str]:
    """Token set from normalized ingredients (split on non-alphanumeric)."""
    tokens: set[str] = set()
    for ingredient in parse_ingredients(ingredients):
        tokens.update(re.findall(r"[a-z0-9]+", ingredient))
    return tokens


def keyword_in_ingredients(ingredients: Iterable[str], keyword: str) -> bool:
    """Check if keyword appears in ingredient text with word-boundary safety."""
    keyword = normalize_text(keyword)
    if not keyword:
        return False

    parsed = parse_ingredients(ingredients)
    blob = " | ".join(parsed)

    if " " in keyword:
        return keyword in blob

    pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
    if re.search(pattern, blob):
        return True

    return keyword in ingredient_tokens(parsed)
