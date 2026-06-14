"""Validated JSON loading for the evaluation test set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

from app.rag.models import GuardrailCategory

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    question: NonEmptyText


class ExpectedDataPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: NonEmptyText
    value: Any


class RagCase(EvaluationCase):
    expected_topic: NonEmptyText
    expected_source_doc: NonEmptyText


class AnalysisCase(EvaluationCase):
    user_id: NonEmptyText
    expected_data_points: tuple[ExpectedDataPoint, ...]


class AgentCase(EvaluationCase):
    user_id: NonEmptyText
    expected_tools: tuple[NonEmptyText, ...]


class GuardrailCase(EvaluationCase):
    expected_category: GuardrailCategory

    @field_validator("expected_category")
    @classmethod
    def require_blocked_category(cls, category: GuardrailCategory) -> GuardrailCategory:
        if category is GuardrailCategory.ALLOWED:
            raise ValueError("Guardrail evaluation cases must expect a blocked category")
        return category


class EvaluationTestSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rag_cases: tuple[RagCase, ...]
    analysis_cases: tuple[AnalysisCase, ...]
    agent_cases: tuple[AgentCase, ...]
    guardrail_cases: tuple[GuardrailCase, ...]

    @property
    def all_cases(self) -> tuple[EvaluationCase, ...]:
        return (
            *self.rag_cases,
            *self.analysis_cases,
            *self.agent_cases,
            *self.guardrail_cases,
        )

    @model_validator(mode="after")
    def validate_cases(self) -> EvaluationTestSet:
        if not self.all_cases:
            raise ValueError("The evaluation test set must contain at least one case")
        case_ids = [case.case_id for case in self.all_cases]
        duplicate_ids = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        if duplicate_ids:
            raise ValueError(
                "Evaluation case IDs must be unique; duplicates: "
                + ", ".join(duplicate_ids)
            )
        return self


def load_test_set(path: Path) -> EvaluationTestSet:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read evaluation test set: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Evaluation test set is not valid JSON at line {exc.lineno}, column {exc.colno}: {path}"
        ) from exc
    return EvaluationTestSet.model_validate(payload)
