"""Coach-assist tool schemas and direct service adapters."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agent.models import (
    AnalyzeHistoryArguments,
    RagSearchArguments,
    ToolExecution,
)
from app.analysis.insight import AnalysisService, get_analysis_service
from app.rag.retrieve import RetrievalService, get_retrieval_service

RAG_SEARCH = "rag_search"
ANALYZE_HISTORY = "analyze_history"


def _tool_schema(
    name: str, description: str, model: type[RagSearchArguments] | type[AnalyzeHistoryArguments]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
            "strict": True,
        },
    }


TOOL_SCHEMAS = [
    _tool_schema(
        RAG_SEARCH,
        "Search grounded fitness knowledge for general training guidance and source citations.",
        RagSearchArguments,
    ),
    _tool_schema(
        ANALYZE_HISTORY,
        "Analyze the authoritative user's workout history and return an insight plus numeric summary.",
        AnalyzeHistoryArguments,
    ),
]


async def rag_search(
    query: str,
    *,
    service: RetrievalService | None = None,
) -> dict[str, Any]:
    """Run grounded retrieval directly without an internal HTTP request."""
    result = await (service or get_retrieval_service()).query(query)
    return {
        "answer": result.answer,
        "sources": [source.__dict__ for source in result.sources],
    }


async def analyze_history(
    user_id: str,
    question: str,
    *,
    service: AnalysisService | None = None,
) -> dict[str, Any]:
    """Run workout-history analysis directly without an internal HTTP request."""
    result = await (service or get_analysis_service()).query(
        user_id=user_id,
        question=question,
    )
    return {"insight": result.insight, "summary": result.summary}


class CoachToolRegistry:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        analysis_service: AnalysisService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    async def execute(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        request_user_id: str,
    ) -> ToolExecution:
        try:
            if name == RAG_SEARCH:
                return await self._rag_search(arguments)
            if name == ANALYZE_HISTORY:
                return await self._analyze_history(arguments, request_user_id)
            return self._error(name, f"Unknown tool: {name}")
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            return self._error(name, f"Invalid tool arguments: {details}")
        except Exception:
            return self._error(name, "Tool execution failed and is temporarily unavailable.")

    def finalize_answer(self, answer: str, executions: list[ToolExecution]) -> str:
        """Apply source attribution owned by the registered tool definitions."""
        successful = [execution for execution in executions if not execution.is_error]
        analysis_used = any(execution.name == ANALYZE_HISTORY for execution in successful)
        rag_results = [execution for execution in successful if execution.name == RAG_SEARCH]
        if not analysis_used or not rag_results:
            return answer

        chunk_ids = list(
            dict.fromkeys(
                source.get("chunk_id")
                for execution in rag_results
                for source in execution.payload.get("sources", [])
                if source.get("chunk_id")
            )
        )
        knowledge_source = (
            ", ".join(f"[{chunk_id}]" for chunk_id in chunk_ids)
            if chunk_ids
            else "fitness knowledge tool (no source chunks returned)"
        )
        return f"{answer}\n\nSources: workout-history analysis; {knowledge_source}."

    async def _rag_search(self, arguments: dict[str, Any]) -> ToolExecution:
        validated = RagSearchArguments.model_validate(arguments)
        payload = await rag_search(validated.query, service=self.retrieval_service)
        return self._success(RAG_SEARCH, payload)

    async def _analyze_history(
        self,
        arguments: dict[str, Any],
        request_user_id: str,
    ) -> ToolExecution:
        validated = AnalyzeHistoryArguments.model_validate(arguments)
        if validated.user_id != request_user_id:
            return self._error(
                ANALYZE_HISTORY,
                "The tool user_id does not match the authoritative request user_id.",
            )
        payload = await analyze_history(
            request_user_id,
            validated.question,
            service=self.analysis_service,
        )
        return self._success(ANALYZE_HISTORY, payload)

    @staticmethod
    def _success(name: str, payload: dict[str, Any]) -> ToolExecution:
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return ToolExecution(name=name, content=content, payload=payload)

    @staticmethod
    def _error(name: str, message: str) -> ToolExecution:
        payload = {"error": message}
        return ToolExecution(
            name=name,
            content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            payload=payload,
            is_error=True,
        )
