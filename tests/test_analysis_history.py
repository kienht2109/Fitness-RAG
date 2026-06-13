import json

import pytest

from app.analysis.history import (
    JsonWorkoutHistoryRepository,
    UserNotFoundError,
    WorkoutHistoryUnavailableError,
)


def _dataset(weight: float = 70) -> dict:
    return {
        "description": "Test workout history",
        "users": {
            "user_a": {
                "name": "Alex",
                "profile": "Intermediate lifter",
                "workouts": [
                    {
                        "date": "2026-01-02",
                        "exercise": "Bench Press",
                        "sets": [{"reps": 8, "weight": weight, "unit": "kg"}],
                    }
                ],
            }
        },
        "edge_cases_notes": [],
    }


def test_json_repository_reads_current_file_for_each_lookup(tmp_path) -> None:
    path = tmp_path / "workout-history.json"
    path.write_text(json.dumps(_dataset(70)), encoding="utf-8")
    repository = JsonWorkoutHistoryRepository(path)

    assert repository.get_user("user_a").workouts[0].sets[0].weight == 70

    path.write_text(json.dumps(_dataset(75)), encoding="utf-8")

    assert repository.get_user("user_a").workouts[0].sets[0].weight == 75


def test_json_repository_rejects_unknown_user(tmp_path) -> None:
    path = tmp_path / "workout-history.json"
    path.write_text(json.dumps(_dataset()), encoding="utf-8")

    with pytest.raises(UserNotFoundError, match="user_b"):
        JsonWorkoutHistoryRepository(path).get_user("user_b")


def test_json_repository_rejects_invalid_source(tmp_path) -> None:
    path = tmp_path / "workout-history.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(WorkoutHistoryUnavailableError):
        JsonWorkoutHistoryRepository(path).get_user("user_a")
