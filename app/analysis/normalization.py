"""Workout-history validation and weight-unit normalization."""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

from app.analysis.constants import EXERCISE_MUSCLE_GROUPS, KG_UNITS, LB_TO_KG, LB_UNITS


def normalize_weight_to_kg(weight: int | float, unit: str) -> float:
    """Convert a non-negative weight to kilograms."""
    value = float(weight)
    if value < 0:
        raise ValueError("Weight cannot be negative")

    normalized_unit = unit.strip().lower()
    if normalized_unit in KG_UNITS:
        return value
    if normalized_unit in LB_UNITS:
        return value * LB_TO_KG
    raise ValueError(f"Unsupported weight unit: {unit!r}")


def muscle_group_for_exercise(exercise: str) -> str:
    """Return the primary muscle group, or ``unknown`` for an unmapped exercise."""
    return EXERCISE_MUSCLE_GROUPS.get(exercise.strip().casefold(), "unknown")


def normalize_workout_history(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate entries and normalize every set weight to kilograms."""
    normalized: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(history):
        workout_date, exercise, sets = _parse_entry(entry, entry_index)
        normalized_sets = [
            _normalize_set(workout_set, entry_index, set_index)
            for set_index, workout_set in enumerate(sets)
        ]
        normalized.append(
            {
                "date": workout_date.isoformat(),
                "exercise": exercise,
                "muscle_group": muscle_group_for_exercise(exercise),
                "sets": normalized_sets,
            }
        )

    return sorted(normalized, key=lambda item: (item["date"], item["exercise"].casefold()))


def _parse_entry(entry: dict[str, Any], entry_index: int) -> tuple[date, str, list[dict[str, Any]]]:
    try:
        workout_date = date.fromisoformat(str(entry["date"]))
        exercise = str(entry["exercise"]).strip()
        sets = entry["sets"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid workout entry at index {entry_index}") from exc

    if not exercise:
        raise ValueError(f"Exercise cannot be empty at entry index {entry_index}")
    if not isinstance(sets, list):
        raise ValueError(f"Sets must be a list at entry index {entry_index}")
    return workout_date, exercise, sets


def _normalize_set(workout_set: dict[str, Any], entry_index: int, set_index: int) -> dict[str, Any]:
    try:
        reps = int(workout_set["reps"])
        original_weight = float(workout_set["weight"])
        original_unit = str(workout_set["unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid set at entry {entry_index}, set {set_index}") from exc

    if reps < 0:
        raise ValueError(f"Reps cannot be negative at entry {entry_index}, set {set_index}")

    return {
        "reps": reps,
        "weight_kg": normalize_weight_to_kg(original_weight, original_unit),
        "original_weight": original_weight,
        "original_unit": original_unit,
    }
