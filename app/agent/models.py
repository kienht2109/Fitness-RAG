"""Validated agent inputs and internal result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

ToolText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
UserId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class RagSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: ToolText


class AnalyzeHistoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UserId
    question: ToolText


@dataclass(frozen=True)
class ToolExecution:
    name: str
    content: str
    payload: dict[str, Any]
    is_error: bool = False
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    answer: str
    tools_used: list[str]
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
