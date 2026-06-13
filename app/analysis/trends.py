"""Orchestration for per-exercise workout trend analysis."""

from __future__ import annotations

from typing import Any, Iterable

from app.analysis.constants import CANONICAL_WEIGHT_UNIT
from app.analysis.normalization import muscle_group_for_exercise, normalize_workout_history
from app.analysis.sessions import (
    build_exercise_sessions,
    detect_deload_dates,
    public_session_point,
)
from app.analysis.trend_metrics import calculate_trend_metrics, progression_pattern
from app.analysis.utils import date_range, detect_training_gaps


def detect_exercise_trends(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Detect date-aware strength and rep trends for each exercise."""
    workouts = normalize_workout_history(history)
    sessions = build_exercise_sessions(workouts)
    trends: dict[str, dict[str, Any]] = {}

    for exercise, points in sorted(sessions.items()):
        points.sort(key=lambda point: point["date"])
        weighted_points = [point for point in points if point["top_weight_kg"] > 0]
        deload_dates = detect_deload_dates(weighted_points)
        fitted_points = [point for point in weighted_points if point["date"] not in deload_dates]

        strength_trend = calculate_trend_metrics(fitted_points, metric="estimated_1rm_kg")
        bodyweight_points = [point for point in points if point["bodyweight_set_count"] > 0]
        rep_trend = calculate_trend_metrics(bodyweight_points, metric="bodyweight_reps")
        units = sorted({unit for point in points for unit in point["original_units"]})

        trends[exercise] = {
            "muscle_group": muscle_group_for_exercise(exercise),
            "session_count": len(points),
            "weighted_session_count": len(weighted_points),
            "bodyweight_session_count": len(bodyweight_points),
            "mixed_units_normalized": len(units) > 1,
            "original_units": units,
            "likely_deload_dates": sorted(deload_dates),
            "strength": strength_trend,
            "bodyweight_reps": rep_trend,
            "progression_pattern": progression_pattern(strength_trend, rep_trend),
            "series": [
                public_session_point(point, point["date"] in deload_dates) for point in points
            ],
        }

    workout_dates = {workout["date"] for workout in workouts}
    return {
        "canonical_weight_unit": CANONICAL_WEIGHT_UNIT,
        "date_range": date_range(workout_dates),
        "gaps": detect_training_gaps(workout_dates),
        "exercises": trends,
    }
