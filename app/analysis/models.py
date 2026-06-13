"""Validated models matching the workout-history JSON format."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class WorkoutSetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reps: int = Field(ge=0)
    weight: float = Field(ge=0)
    unit: Literal["kg", "lb"]


class WorkoutRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date
    exercise: NonEmptyString
    sets: list[WorkoutSetRecord]


class AnalysisInsight(BaseModel):
    insight: str = Field(
        min_length=1,
        description="A concise workout-history insight grounded only in the supplied summary.",
    )
