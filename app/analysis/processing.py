"""Public facade for deterministic workout-history processing."""

from __future__ import annotations

from typing import Any, Iterable

from app.analysis.aggregation import aggregate_workout_data
from app.analysis.constants import EXERCISE_MUSCLE_GROUPS, LB_TO_KG
from app.analysis.normalization import (
    muscle_group_for_exercise,
    normalize_weight_to_kg,
    normalize_workout_history,
)
from app.analysis.trends import detect_exercise_trends

__all__ = [
    "EXERCISE_MUSCLE_GROUPS",
    "LB_TO_KG",
    "aggregate_workout_data",
    "build_workout_summary",
    "detect_exercise_trends",
    "muscle_group_for_exercise",
    "normalize_weight_to_kg",
    "normalize_workout_history",
]


def build_workout_summary(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build the complete deterministic aggregation and trend summary."""
    materialized = list(history)
    return {
        "aggregation": aggregate_workout_data(materialized),
        "trends": detect_exercise_trends(materialized),
    }
