"""Domain constants used by deterministic workout analysis."""

LB_TO_KG = 0.45359237
CANONICAL_WEIGHT_UNIT = "kg"

KG_UNITS = frozenset({"kg", "kgs", "kilogram", "kilograms"})
LB_UNITS = frozenset({"lb", "lbs", "pound", "pounds"})

# Each exercise has one primary group so muscle-group volume is not double counted.
EXERCISE_MUSCLE_GROUPS: dict[str, str] = {
    "barbell row": "back",
    "bench press": "chest",
    "bicep curl": "arms",
    "deadlift": "posterior_chain",
    "face pull": "back",
    "incline dumbbell press": "chest",
    "lateral raise": "shoulders",
    "leg press": "quadriceps",
    "overhead press": "shoulders",
    "pull-up": "back",
    "pull up": "back",
    "romanian deadlift": "posterior_chain",
    "squat": "quadriceps",
    "tricep pushdown": "arms",
}

DELOAD_STRENGTH_RATIO = 0.85
DELOAD_VOLUME_RATIO = 0.70
DELOAD_RECOVERY_RATIO = 0.95
DELOAD_MAX_RECOVERY_DAYS = 21
MINIMUM_TRAINING_GAP_DAYS = 14

TREND_PERCENT_CHANGE_THRESHOLD = 2.5
TREND_WEEKLY_SLOPE_THRESHOLD = 0.1
STRUCTURED_PROGRESSION_R_SQUARED = 0.5
