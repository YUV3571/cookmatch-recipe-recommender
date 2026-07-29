"""Meal intent tag mappings for Stage 3 content filtering."""

MEAL_INTENT_TAGS = {
    "dessert": "desserts",
    "desserts": "desserts",
    "cake": "cakes",
    "cakes": "cakes",
    "main": "main-dish",
    "main-dish": "main-dish",
    "appetizer": "appetizers",
    "appetizers": "appetizers",
    "breakfast": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner",
    "snack": "snacks",
    "snacks": "snacks",
    "soup": "soups",
    "soups": "soups",
    "salad": "salad",
    "pasta": "pasta",
    "bread": "bread",
}

SUPPORTED_MEAL_INTENTS = tuple(sorted(set(MEAL_INTENT_TAGS.keys())))
