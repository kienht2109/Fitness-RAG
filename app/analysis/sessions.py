"""Exercise-session series construction and deload detection."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

from app.analysis.constants import (
    DELOAD_MAX_RECOVERY_DAYS,
    DELOAD_RECOVERY_RATIO,
    DELOAD_STRENGTH_RATIO,
    DELOAD_VOLUME_RATIO,
)


def build_exercise_sessions(
    workouts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    display_names: dict[str, str] = {}
    for workout in workouts:
        normalized_name = workout["exercise"].casefold()
        display_names.setdefault(normalized_name, workout["exercise"])
        key = (normalized_name, workout["date"])
        point = buckets.setdefault(key, _empty_session(workout["date"]))
        for workout_set in workout["sets"]:
            _add_set_to_session(point, workout_set)

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (normalized_name, _), point in buckets.items():
        result[display_names[normalized_name]].append(point)
    return dict(result)


def detect_deload_dates(points: list[dict[str, Any]]) -> set[str]:
    deload_dates: set[str] = set()
    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        current = points[index]
        following = points[index + 1]
        previous_metric = previous["estimated_1rm_kg"]
        if previous_metric <= 0:
            continue

        strength_ratio = current["estimated_1rm_kg"] / previous_metric
        recovery_ratio = following["estimated_1rm_kg"] / previous_metric
        volume_ratio = current["volume_kg"] / previous["volume_kg"] if previous["volume_kg"] else 1
        recovery_days = (
            date.fromisoformat(following["date"]) - date.fromisoformat(current["date"])
        ).days

        if (
            strength_ratio <= DELOAD_STRENGTH_RATIO
            and volume_ratio <= DELOAD_VOLUME_RATIO
            and recovery_ratio >= DELOAD_RECOVERY_RATIO
            and recovery_days <= DELOAD_MAX_RECOVERY_DAYS
        ):
            deload_dates.add(current["date"])
    return deload_dates


def public_session_point(point: dict[str, Any], is_deload: bool) -> dict[str, Any]:
    return {
        "date": point["date"],
        "top_weight_kg": round(point["top_weight_kg"], 2),
        "estimated_1rm_kg": round(point["estimated_1rm_kg"], 2),
        "total_reps": point["total_reps"],
        "volume_kg": round(point["volume_kg"], 2),
        "bodyweight_reps": point["bodyweight_reps"],
        "bodyweight_set_count": point["bodyweight_set_count"],
        "is_likely_deload": is_deload,
    }


def _empty_session(workout_date: str) -> dict[str, Any]:
    return {
        "date": workout_date,
        "top_weight_kg": 0.0,
        "estimated_1rm_kg": 0.0,
        "total_reps": 0,
        "volume_kg": 0.0,
        "bodyweight_reps": 0,
        "bodyweight_set_count": 0,
        "original_units": set(),
    }


def _add_set_to_session(point: dict[str, Any], workout_set: dict[str, Any]) -> None:
    reps = workout_set["reps"]
    weight_kg = workout_set["weight_kg"]
    point["total_reps"] += reps
    point["volume_kg"] += reps * weight_kg
    point["original_units"].add(workout_set["original_unit"].strip().lower())
    if weight_kg == 0:
        point["bodyweight_reps"] += reps
        point["bodyweight_set_count"] += 1
        return

    point["top_weight_kg"] = max(point["top_weight_kg"], weight_kg)
    point["estimated_1rm_kg"] = max(point["estimated_1rm_kg"], weight_kg * (1 + reps / 30))
