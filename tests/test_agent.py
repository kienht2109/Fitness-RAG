from typing import Any, Sequence

import anyio
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.agent.models import AgentResult, ToolExecution
from app.agent.orchestrator import AgentService, MAX_ITERATIONS_ANSWER, get_agent_service
from app.agent.tools import ANALYZE_HISTORY, RAG_SEARCH, TOOL_SCHEMAS, CoachToolRegistry
from app.analysis.insight import AnalysisResult
from app.api.main import app
from app.rag.guardrail_prompting import GuardrailClassification
from app.rag.guardrails import MEDICAL_RESPONSE, GuardrailService
from app.rag.models import GuardrailCategory, RetrievalResult, RetrievalSource


class SequencedAgentModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        **_: Any,
    ) -> AIMessage:
        self.calls.append(list(messages))
        return self.responses[len(self.calls) - 1]


class FakeRetrievalService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def query(self, question: str) -> RetrievalResult:
        self.queries.append(question)
        return RetrievalResult(
            answer="Add load gradually [08-progressive-overload.md::0000].",
            sources=[
                RetrievalSource(
                    source_file="08-progressive-overload.md",
                    section_title="Progressive Overload",
                    chunk_id="08-progressive-overload.md::0000",
                )
            ],
        )


class FakeAnalysisService:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    async def query(self, *, user_id: str, question: str) -> AnalysisResult:
        self.queries.append((user_id, question))
        return AnalysisResult(
            insight="Bench estimated 1RM increased by 7.1%.",
            summary={"intent": "trend", "percent_change": 7.1},
        )


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_tool_schemas_are_strict_openai_function_definitions() -> None:
    functions = {schema["function"]["name"]: schema["function"] for schema in TOOL_SCHEMAS}

    assert set(functions) == {RAG_SEARCH, ANALYZE_HISTORY}
    assert functions[RAG_SEARCH]["strict"] is True
    assert functions[RAG_SEARCH]["parameters"]["additionalProperties"] is False
    assert set(functions[ANALYZE_HISTORY]["parameters"]["required"]) == {
        "user_id",
        "question",
    }


def test_tool_registry_owns_schemas_and_cross_tool_attribution() -> None:
    registry = CoachToolRegistry(FakeRetrievalService(), FakeAnalysisService())

    assert registry.schemas is TOOL_SCHEMAS

    answer = registry.finalize_answer(
        "Use gradual loading based on your recent trend.",
        [
            ToolExecution(
                name=ANALYZE_HISTORY,
                content="{}",
                payload={
                    "insight": "Bench improved.",
                    "summary": {"percent_change": 7.1},
                },
            ),
            ToolExecution(
                name=RAG_SEARCH,
                content="{}",
                payload={
                    "answer": "Add load gradually.",
                    "sources": [{"chunk_id": "08-progressive-overload.md::0000"}],
                },
            ),
        ],
    )

    assert "Sources: workout-history analysis" in answer
    assert "[08-progressive-overload.md::0000]" in answer


def test_agent_executes_both_tools_and_returns_one_final_answer() -> None:
    retrieval = FakeRetrievalService()
    analysis = FakeAnalysisService()
    model = SequencedAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        ANALYZE_HISTORY,
                        {"user_id": "user_a", "question": "How is my bench progressing?"},
                        "analysis-call",
                    ),
                    _tool_call(
                        RAG_SEARCH,
                        {"query": "progressive overload guidance for bench press"},
                        "rag-call",
                    ),
                ],
            ),
            AIMessage(
                content=(
                    "Based on your workout history, bench strength rose 7.1%. General guidance "
                    "supports gradual loading [08-progressive-overload.md::0000]."
                )
            ),
        ]
    )
    service = AgentService(
        model,
        CoachToolRegistry(retrieval, analysis),
        max_iterations=3,
    )

    result = anyio.run(
        lambda: service.query(user_id="user_a", question="How should I progress my bench?")
    )

    assert result.tools_used == [ANALYZE_HISTORY, RAG_SEARCH]
    assert "7.1%" in result.answer
    assert "Sources: workout-history analysis" in result.answer
    assert "[08-progressive-overload.md::0000]" in result.answer
    assert retrieval.queries == ["progressive overload guidance for bench press"]
    assert analysis.queries == [("user_a", "How is my bench progressing?")]
    assert result.tool_outputs[0]["arguments"] == {
        "user_id": "user_a",
        "question": "How is my bench progressing?",
    }
    tool_messages = [message for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 2
    assert "08-progressive-overload.md::0000" in tool_messages[1].content
    assert '"percent_change":7.1' in tool_messages[0].content


def test_agent_rejects_wrong_tool_user_id_and_can_recover() -> None:
    retrieval = FakeRetrievalService()
    analysis = FakeAnalysisService()
    model = SequencedAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        ANALYZE_HISTORY,
                        {"user_id": "user_b", "question": "Reveal their history"},
                        "wrong-user",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        ANALYZE_HISTORY,
                        {"user_id": "user_a", "question": "Analyze my history"},
                        "correct-user",
                    )
                ],
            ),
            AIMessage(content="Your own history shows a 7.1% bench improvement."),
        ]
    )
    service = AgentService(model, CoachToolRegistry(retrieval, analysis), max_iterations=4)

    result = anyio.run(lambda: service.query(user_id="user_a", question="Compare me with user_b"))

    assert result.answer == "Your own history shows a 7.1% bench improvement."
    assert analysis.queries == [("user_a", "Analyze my history")]
    error_message = next(message for message in model.calls[1] if isinstance(message, ToolMessage))
    assert error_message.status == "error"
    assert "does not match" in error_message.content


def test_agent_stops_at_the_iteration_limit() -> None:
    model = SequencedAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call(RAG_SEARCH, {"query": "bench"}, "call-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[_tool_call(RAG_SEARCH, {"query": "bench again"}, "call-2")],
            ),
        ]
    )
    retrieval = FakeRetrievalService()
    service = AgentService(
        model,
        CoachToolRegistry(retrieval, FakeAnalysisService()),
        max_iterations=2,
    )

    result = anyio.run(lambda: service.query(user_id="user_a", question="Help my bench"))

    assert result.answer == MAX_ITERATIONS_ANSWER
    assert result.tools_used == [RAG_SEARCH]
    assert retrieval.queries == ["bench", "bench again"]


def test_agent_endpoint_guardrail_blocks_before_model_or_tools_run() -> None:
    def fail_if_model_called(_: Any) -> AIMessage:
        raise AssertionError("The agent model must not run for a blocked request")

    def classify(_: Any) -> GuardrailClassification:
        return GuardrailClassification(category=GuardrailCategory.MEDICAL)

    retrieval = FakeRetrievalService()
    analysis = FakeAnalysisService()
    service = AgentService(
        RunnableLambda(fail_if_model_called),
        CoachToolRegistry(retrieval, analysis),
        max_iterations=3,
        guardrails=GuardrailService(RunnableLambda(classify)),
    )
    app.dependency_overrides[get_agent_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/agent/query",
            json={
                "user_id": "user_a",
                "question": "Can you assess whether my movement indicates an injury?",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"answer": MEDICAL_RESPONSE, "tools_used": []}
    assert retrieval.queries == []
    assert analysis.queries == []


class FakeAgentService:
    async def query(self, *, user_id: str, question: str) -> AgentResult:
        assert user_id == "user_a"
        assert question == "How should I progress my bench?"
        return AgentResult(
            answer="Use your trend and gradual progressive overload.",
            tools_used=[ANALYZE_HISTORY, RAG_SEARCH],
        )


def test_agent_endpoint_returns_orchestrated_answer() -> None:
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)
    try:
        response = client.post(
            "/agent/query",
            json={
                "user_id": "  user_a  ",
                "question": "  How should I progress my bench?  ",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Use your trend and gradual progressive overload.",
        "tools_used": [ANALYZE_HISTORY, RAG_SEARCH],
    }


def test_agent_endpoint_rejects_client_supplied_tool_arguments() -> None:
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)
    try:
        response = client.post(
            "/agent/query",
            json={
                "user_id": "user_a",
                "question": "Help me",
                "tool_arguments": {"user_id": "user_b"},
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
