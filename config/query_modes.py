"""Weight presets for three-panel demo query modes."""

QUERY_MODES: dict[str, dict] = {
    "pantry": {
        "weights": {"cf": 0.0, "pantry": 1.0, "time": 0.0, "intent": 0.0},
        "hard_time_cap": False,
        "label": "Pantry",
        "description": "What can I cook with what I have?",
    },
    "time": {
        "weights": {"cf": 0.0, "pantry": 0.0, "time": 1.0, "intent": 0.0},
        "hard_time_cap": True,
        "label": "Time",
        "description": "Recipes within my time budget.",
    },
    "intent": {
        "weights": {"cf": 0.0, "pantry": 0.0, "time": 0.0, "intent": 1.0},
        "hard_time_cap": False,
        "label": "Intent",
        "description": "Recipes matching my meal goal.",
    },
    "combined": {
        "weights": None,
        "hard_time_cap": False,
        "label": "Combined",
        "description": "Personalized blend of all signals.",
    },
}
