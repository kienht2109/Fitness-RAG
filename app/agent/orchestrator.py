"""Bounded tool-calling loop for the coach-assist agent."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Protocol, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.models import AgentResult, ToolExecution
from app.agent.prompting import build_initial_messages
from app.agent.tools import TOOL_SCHEMAS, CoachToolRegistry
from app.analysis.insight import create_analysis_service
from app.core.ai import create_chat_model
from app.core.config import Settings, get_settings
from app.rag.retrieve import create_retrieval_service

MAX_ITERATIONS_ANSWER = (
    "I could not complete the coach-assist request within the tool-call limit. "
    "Please narrow the question and try again."
)
EMPTY_AGENT_ANSWER = "I could not produce a coach-assist answer for this request."


class AgentModel(Protocol):
    async def ainvoke(self, input: Sequence[BaseMessage], **kwargs: Any) -> AIMessage: ...


class AgentService:
    def __init__(
        self,
        agent_model: AgentModel,
        tool_registry: CoachToolRegistry,
        max_iterations: int,
    ) -> None:
        self.agent_model = agent_model
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    async def query(self, *, user_id: str, question: str) -> AgentResult:
        normalized_user_id = user_id.strip()
        normalized_question = question.strip()
        if not normalized_user_id:
            raise ValueError("user_id must not be empty")
        if not normalized_question:
            raise ValueError("Question must not be empty")

        messages: list[BaseMessage] = build_initial_messages(
            user_id=normalized_user_id,
            question=normalized_question,
        )
        tools_used: list[str] = []
        observations: list[ToolExecution] = []

        for _ in range(self.max_iterations):
            response = await self.agent_model.ainvoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError("The agent model must return an AIMessage")
            messages.append(response)

            if not response.tool_calls:
                return AgentResult(
                    answer=_ensure_attribution(
                        _message_text(response) or EMPTY_AGENT_ANSWER,
                        observations,
                    ),
                    tools_used=tools_used,
                )

            executions = await asyncio.gather(
                *(
                    self.tool_registry.execute(
                        name=tool_call["name"],
                        arguments=tool_call.get("args", {}),
                        request_user_id=normalized_user_id,
                    )
                    for tool_call in response.tool_calls
                )
            )
            for tool_call, execution in zip(response.tool_calls, executions, strict=True):
                observations.append(execution)
                if execution.name not in tools_used:
                    tools_used.append(execution.name)
                messages.append(_tool_message(tool_call["id"], execution))

        return AgentResult(answer=MAX_ITERATIONS_ANSWER, tools_used=tools_used)


def _tool_message(tool_call_id: str, execution: ToolExecution) -> ToolMessage:
    return ToolMessage(
        content=execution.content,
        tool_call_id=tool_call_id,
        name=execution.name,
        status="error" if execution.is_error else "success",
    )


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part.strip() for part in parts if part.strip())


def _ensure_attribution(answer: str, observations: list[ToolExecution]) -> str:
    successful = [observation for observation in observations if not observation.is_error]
    analysis_used = any(observation.name == "analyze_history" for observation in successful)
    rag_results = [observation for observation in successful if observation.name == "rag_search"]
    if not analysis_used or not rag_results:
        return answer

    chunk_ids = list(
        dict.fromkeys(
            source.get("chunk_id")
            for observation in rag_results
            for source in observation.payload.get("sources", [])
            if source.get("chunk_id")
        )
    )
    knowledge_source = (
        ", ".join(f"[{chunk_id}]" for chunk_id in chunk_ids)
        if chunk_ids
        else "fitness knowledge tool (no source chunks returned)"
    )
    return f"{answer}\n\nSources: workout-history analysis; {knowledge_source}."


def create_agent_service(settings: Settings | None = None) -> AgentService:
    settings = settings or get_settings()
    chat_model = create_chat_model(settings, model=settings.openai_agent_model)
    agent_model = chat_model.bind_tools(
        TOOL_SCHEMAS,
        strict=True,
        parallel_tool_calls=True,
    )
    return AgentService(
        agent_model=agent_model,
        tool_registry=CoachToolRegistry(
            retrieval_service=create_retrieval_service(settings),
            analysis_service=create_analysis_service(settings),
        ),
        max_iterations=settings.agent_max_iterations,
    )


@lru_cache
def get_agent_service() -> AgentService:
    return create_agent_service()
