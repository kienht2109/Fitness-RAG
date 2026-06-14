"""Validated JSON loading for the evaluation test set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from app.rag.models import GuardrailCategory

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    question: NonEmptyText
    edge_cases_covered: tuple[NonEmptyText, ...] = Field(min_length=1)
    reference_data: dict[str, Any] = Field(default_factory=dict)
    correct_answer_criteria: tuple[NonEmptyText, ...] = Field(min_length=1)
    failure_modes: tuple[NonEmptyText, ...] = Field(min_length=1)


class ExpectedDataPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: NonEmptyText
    value: Any
    description: NonEmptyText | None = None


class ExpectedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: NonEmptyText
    required: bool = True
    argument_criteria: tuple[NonEmptyText, ...] = ()
    expected_result: NonEmptyText | None = None


class RagCase(EvaluationCase):
    expected_topic: NonEmptyText
    expected_source_doc: NonEmptyText


class AnalysisCase(EvaluationCase):
    user_id: NonEmptyText
    expected_data_points: tuple[ExpectedDataPoint, ...]


class AgentCase(EvaluationCase):
    user_id: NonEmptyText
    expected_tool_calls: tuple[ExpectedToolCall, ...] = Field(min_length=1)
    tool_order_strict: bool = False
    expected_data_points: tuple[ExpectedDataPoint, ...] = ()


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
    coverage_summary: dict[NonEmptyText, tuple[NonEmptyText, ...]]

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
        known_case_ids = set(case_ids)
        for coverage_name, covered_case_ids in self.coverage_summary.items():
            if not covered_case_ids:
                raise ValueError(f"Coverage entry must reference at least one case: {coverage_name}")
            unknown_case_ids = sorted(set(covered_case_ids) - known_case_ids)
            if unknown_case_ids:
                raise ValueError(
                    f"Coverage entry {coverage_name!r} references unknown cases: "
                    + ", ".join(unknown_case_ids)
                )
        for case in self.all_cases:
            missing_coverage = [
                edge_case
                for edge_case in case.edge_cases_covered
                if case.case_id not in self.coverage_summary.get(edge_case, ())
            ]
            if missing_coverage:
                raise ValueError(
                    f"Case {case.case_id!r} is missing from coverage_summary entries: "
                    + ", ".join(missing_coverage)
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
