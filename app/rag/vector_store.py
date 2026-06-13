from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.chroma import create_chroma_client
from app.core.config import Settings
from app.rag.chunking import MANAGED_BY
from app.rag.models import KnowledgeChunk


T = TypeVar("T")


def create_vector_store(
    settings: Settings,
    embeddings: Embeddings,
    chroma_client: Any | None = None,
) -> Chroma:
    client = chroma_client or create_chroma_client(settings)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={
            "description": "Header-aware fitness knowledge chunks",
            "embedding_model": settings.openai_embedding_model,
            "managed_by": MANAGED_BY,
        },
    )
    collection_model = (collection.metadata or {}).get("embedding_model")
    if collection_model and collection_model != settings.openai_embedding_model:
        raise RuntimeError(
            f"Collection {settings.chroma_collection!r} uses embedding model "
            f"{collection_model!r}, not {settings.openai_embedding_model!r}. "
            "Use a new collection name or recreate the collection before ingesting."
        )

    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )


def _batched(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def store_chunks(
    chunks: Sequence[KnowledgeChunk],
    vector_store: Chroma,
    batch_size: int,
) -> int:
    existing = vector_store.get(where={"managed_by": MANAGED_BY}, include=["metadatas"])
    existing_ids = set(existing.get("ids") or [])

    for batch in _batched(chunks, batch_size):
        documents = [
            Document(page_content=chunk.text, metadata=chunk.metadata, id=chunk.chunk_id)
            for chunk in batch
        ]
        vector_store.add_documents(
            documents=documents,
            ids=[chunk.chunk_id for chunk in batch],
        )

    stale_ids = sorted(existing_ids - {chunk.chunk_id for chunk in chunks})
    if stale_ids:
        vector_store.delete(ids=stale_ids)
    return len(stale_ids)
