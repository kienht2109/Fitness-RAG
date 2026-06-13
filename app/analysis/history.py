"""Validated access to the JSON workout-history data source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.analysis.models import UserWorkoutHistory, WorkoutHistoryDataset


class WorkoutHistoryError(Exception):
    """Base error for workout-history data-source failures."""


class WorkoutHistoryUnavailableError(WorkoutHistoryError):
    """Raised when the configured data source cannot be read or validated."""


class UserNotFoundError(WorkoutHistoryError):
    """Raised when a user ID does not exist in the data source."""


class WorkoutHistoryRepository(Protocol):
    def get_user(self, user_id: str) -> UserWorkoutHistory: ...


class JsonWorkoutHistoryRepository:
    """Read and validate the current JSON data source for each lookup."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_user(self, user_id: str) -> UserWorkoutHistory:
        dataset = self._load_dataset()
        try:
            return dataset.users[user_id]
        except KeyError as exc:
            raise UserNotFoundError(f"Unknown user_id: {user_id}") from exc

    def _load_dataset(self) -> WorkoutHistoryDataset:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return WorkoutHistoryDataset.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise WorkoutHistoryUnavailableError(
                f"Unable to load valid workout history from {self.path}"
            ) from exc
