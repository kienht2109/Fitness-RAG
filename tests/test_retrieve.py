from typing import Any

import anyio
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.api.main import app
from app.rag.models import RetrievalResult, RetrievalSource
from app.rag.retrieve import NO_CONTEXT_ANSWER, RetrievalService, get_retrieval_service


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int = 4, **_: Any) -> list[Document]:
        self.queries.append((query, k))
        return self.documents[:k]


def test_retrieval_grounds_prompt_and_returns_document_sources() -> None:
    documents = [
        Document(
            id="01-bench-press.md::0001",
            page_content="Always use a spotter for sets near failure.",
            metadata={
                "source_file": "01-bench-press.md",
                "section_title": "Programming Recommendations | Safety",
                "chunk_id": "01-bench-press.md::0001",
            },
        ),
        Document(
            id="01-bench-press.md::0000",
            page_content="Keep the shoulder blades retracted and depressed.",
            metadata={
                "source_file": "01-bench-press.md",
                "section_title": "Proper Form",
                "chunk_id": "01-bench-press.md::0000",
            },
        ),
    ]
    prompts: list[str] = []

    def answer(prompt: Any) -> AIMessage:
        prompts.append(str(prompt))
        return AIMessage(
            content=(
                "Use a spotter near failure [01-bench-press.md::0001] and keep your shoulder "
                "blades retracted [01-bench-press.md::0000]."
            )
        )

    vector_store = FakeVectorStore(documents)
    service = RetrievalService(vector_store, RunnableLambda(answer), top_k=2)

    result = anyio.run(service.query, "How can I bench safely?")

    assert vector_store.queries == [("How can I bench safely?", 2)]
    assert "01-bench-press.md::0001" in prompts[0]
    assert "Always use a spotter" in prompts[0]
    assert result.sources == [
        RetrievalSource(
            source_file="01-bench-press.md",
            section_title="Programming Recommendations | Safety",
            chunk_id="01-bench-press.md::0001",
        ),
        RetrievalSource(
            source_file="01-bench-press.md",
            section_title="Proper Form",
            chunk_id="01-bench-press.md::0000",
        ),
    ]


def test_retrieval_skips_generation_when_no_context_is_found() -> None:
    def fail_if_called(_: Any) -> AIMessage:
        raise AssertionError("The chat model must not be called without context")

    service = RetrievalService(
        FakeVectorStore([]),
        RunnableLambda(fail_if_called),
        top_k=5,
    )

    result = anyio.run(service.query, "Unknown topic")

    assert result == RetrievalResult(answer=NO_CONTEXT_ANSWER, sources=[])


class FakeRetrievalService:
    async def query(self, question: str) -> RetrievalResult:
        assert question == "How should I warm up?"
        return RetrievalResult(
            answer="Use a general and exercise-specific warm-up [19-warm-up-cooldown.md::0000].",
            sources=[
                RetrievalSource(
                    source_file="19-warm-up-cooldown.md",
                    section_title="Why Warm Up? | Warm-Up Protocol",
                    chunk_id="19-warm-up-cooldown.md::0000",
                )
            ],
        )


def test_rag_endpoint_returns_retrieval_result() -> None:
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    client = TestClient(app)
    try:
        response = client.post("/rag/query", json={"question": "  How should I warm up?  "})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "Use a general and exercise-specific warm-up "
            "[19-warm-up-cooldown.md::0000]."
        ),
        "sources": [
            {
                "source_file": "19-warm-up-cooldown.md",
                "section_title": "Why Warm Up? | Warm-Up Protocol",
                "chunk_id": "19-warm-up-cooldown.md::0000",
            }
        ],
    }


def test_rag_endpoint_rejects_blank_questions() -> None:
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    client = TestClient(app)
    try:
        response = client.post("/rag/query", json={"question": "   "})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
