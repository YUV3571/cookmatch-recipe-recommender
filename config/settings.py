"""Project-wide settings for the recipe recommender."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_SLUG = "shuyangli94/food-com-recipes-and-user-interactions"

# Stage 1 scope (v1)
SUPPORTED_DIETS = ("vegetarian", "vegan")
SUPPORTED_ALLERGENS = ("nuts", "dairy", "gluten")

# Primary tables used across stages
RECIPES_FILE = "RAW_recipes.csv"
INTERACTIONS_FILE = "RAW_interactions.csv"
INTERACTIONS_TRAIN_FILE = "interactions_train.csv"
INTERACTIONS_VALIDATION_FILE = "interactions_validation.csv"
INTERACTIONS_TEST_FILE = "interactions_test.csv"

# Stage 2 defaults
CF_FACTORS = 20
POPULARITY_MIN_PRIOR = 50
