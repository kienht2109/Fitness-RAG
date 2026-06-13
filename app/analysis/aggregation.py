"""Volume, repetition, and training-frequency aggregation."""

from __future__ import annotations

from typing import Any, Iterable

from app.analysis.constants import CANONICAL_WEIGHT_UNIT
from app.analysis.normalization import normalize_workout_history
from app.analysis.utils import date_range


def aggregate_workout_data(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate normalized volume, repetitions, sets, and training frequency."""
    workouts = normalize_workout_history(history)
    exercise_buckets: dict[str, dict[str, Any]] = {}
    group_buckets: dict[str, dict[str, Any]] = {}
    all_dates: set[str] = set()
    units_seen: set[str] = set()

    for workout in workouts:
        exercise = workout["exercise"]
        group = workout["muscle_group"]
        workout_date = workout["date"]
        all_dates.add(workout_date)

        exercise_stats = exercise_buckets.setdefault(exercise, _empty_aggregate(group))
        group_stats = group_buckets.setdefault(group, _empty_aggregate(group))
        exercise_stats["_dates"].add(workout_date)
        group_stats["_dates"].add(workout_date)

        for workout_set in workout["sets"]:
            reps = workout_set["reps"]
            weight_kg = workout_set["weight_kg"]
            units_seen.add(workout_set["original_unit"].strip().lower())
            _add_set(exercise_stats, reps, weight_kg)
            _add_set(group_stats, reps, weight_kg)

    return {
        "canonical_weight_unit": CANONICAL_WEIGHT_UNIT,
        "date_range": date_range(all_dates),
        "training_day_count": len(all_dates),
        "mixed_units_normalized": len(units_seen) > 1,
        "original_units": sorted(units_seen),
        "by_exercise": {
            name: _finalize_aggregate(stats) for name, stats in sorted(exercise_buckets.items())
        },
        "by_muscle_group": {
            name: _finalize_aggregate(stats) for name, stats in sorted(group_buckets.items())
        },
    }


def _empty_aggregate(muscle_group: str) -> dict[str, Any]:
    return {
        "muscle_group": muscle_group,
        "_dates": set(),
        "set_count": 0,
        "rep_count": 0,
        "weight_volume_kg": 0.0,
        "bodyweight_set_count": 0,
        "bodyweight_rep_count": 0,
    }


def _add_set(stats: dict[str, Any], reps: int, weight_kg: float) -> None:
    stats["set_count"] += 1
    stats["rep_count"] += reps
    stats["weight_volume_kg"] += reps * weight_kg
    if weight_kg == 0:
        stats["bodyweight_set_count"] += 1
        stats["bodyweight_rep_count"] += reps


def _finalize_aggregate(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "muscle_group": stats["muscle_group"],
        "session_count": len(stats["_dates"]),
        "set_count": stats["set_count"],
        "rep_count": stats["rep_count"],
        "weight_volume_kg": round(stats["weight_volume_kg"], 2),
        "bodyweight_set_count": stats["bodyweight_set_count"],
        "bodyweight_rep_count": stats["bodyweight_rep_count"],
    }
