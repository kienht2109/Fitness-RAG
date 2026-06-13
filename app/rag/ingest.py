import argparse
import json
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.ai import create_embeddings
from app.core.config import Settings, get_settings
from app.rag.chunking import build_chunks
from app.rag.documents import load_markdown_files
from app.rag.models import ChunkingConfig, IngestionReport, KnowledgeChunk
from app.rag.vector_store import create_vector_store, store_chunks


def chunking_config(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(
        min_tokens=settings.rag_chunk_min_tokens,
        target_tokens=settings.rag_chunk_target_tokens,
        max_tokens=settings.rag_chunk_max_tokens,
        overlap_tokens=settings.rag_chunk_overlap_tokens,
    )


def ingest_knowledge_base(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    chroma_client: Any | None = None,
) -> IngestionReport:
    settings = settings or get_settings()
    source_files = load_markdown_files(settings.knowledge_base_dir)
    chunks = build_chunks(settings.knowledge_base_dir, chunking_config(settings))
    chunks = [
        KnowledgeChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata={**chunk.metadata, "embedding_model": settings.openai_embedding_model},
        )
        for chunk in chunks
    ]
    embedding_model = embeddings or create_embeddings(settings)
    vector_store = create_vector_store(settings, embedding_model, chroma_client)
    stale_chunks_deleted = store_chunks(
        chunks,
        vector_store,
        batch_size=settings.rag_embedding_batch_size,
    )
    return IngestionReport(
        source_files=len(source_files),
        chunks_upserted=len(chunks),
        stale_chunks_deleted=stale_chunks_deleted,
    )


def dry_run_summary(settings: Settings) -> dict[str, int | float]:
    chunks = build_chunks(settings.knowledge_base_dir, chunking_config(settings))
    token_counts = [int(chunk.metadata["token_count"]) for chunk in chunks]
    return {
        "source_files": len(load_markdown_files(settings.knowledge_base_dir)),
        "chunks": len(chunks),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "average_tokens": round(sum(token_counts) / len(token_counts), 1),
        "first_chunks": [chunk.text for chunk in chunks[:3]],
        "first_chunks_meta": [chunk.metadata for chunk in chunks[:3]]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the fitness knowledge base into Chroma")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and summarize chunks without calling OpenAI or Chroma",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    output = dry_run_summary(settings) if args.dry_run else ingest_knowledge_base(settings).__dict__
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
