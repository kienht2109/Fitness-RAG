from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from anyio import to_thread
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from app.core.ai import create_chat_model, create_embeddings
from app.core.config import Settings, get_settings
from app.rag.guardrails import GuardrailService, create_guardrail_service
from app.rag.models import RetrievalResult, RetrievalSource
from app.rag.prompting import NO_CONTEXT_ANSWER, build_answer_chain, format_retrieval_context
from app.rag.vector_store import create_vector_store


class SimilaritySearchStore(Protocol):
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]: ...


def sources_from_documents(documents: list[Document]) -> list[RetrievalSource]:
    sources: list[RetrievalSource] = []
    seen: set[tuple[str, str, str | None]] = set()
    for document in documents:
        metadata = document.metadata
        source = RetrievalSource(
            source_file=str(metadata.get("source_file", "unknown")),
            section_title=str(metadata.get("section_title", "unknown")),
            chunk_id=str(metadata["chunk_id"]) if metadata.get("chunk_id") else document.id,
        )
        key = (source.source_file, source.section_title, source.chunk_id)
        if key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


class RetrievalService:
    def __init__(
        self,
        vector_store: SimilaritySearchStore,
        chat_model: Runnable[Any, Any],
        top_k: int,
        guardrails: GuardrailService | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.guardrails = guardrails
        self.answer_chain = build_answer_chain(chat_model)

    async def query(self, question: str) -> RetrievalResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        if self.guardrails is not None:
            decision = await self.guardrails.evaluate(question)
            if decision.blocked:
                return RetrievalResult(answer=decision.response or "", sources=[])

        documents = await to_thread.run_sync(
            lambda: self.vector_store.similarity_search(question, k=self.top_k)
        )
        if not documents:
            return RetrievalResult(answer=NO_CONTEXT_ANSWER, sources=[])

        context = format_retrieval_context(documents)
        answer = await self.answer_chain.ainvoke({"question": question, "context": context})
        return RetrievalResult(
            answer=answer.strip(),
            sources=sources_from_documents(documents),
            context=context,
        )


def create_retrieval_service(settings: Settings | None = None) -> RetrievalService:
    settings = settings or get_settings()
    embeddings = create_embeddings(settings)
    chat_model = create_chat_model(settings)
    return RetrievalService(
        vector_store=create_vector_store(settings, embeddings),
        chat_model=chat_model,
        top_k=settings.rag_top_k,
        guardrails=create_guardrail_service(chat_model),
    )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return create_retrieval_service()
