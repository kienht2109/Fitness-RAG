from pathlib import Path

from langchain_core.documents import Document

from app.rag.chunking import MANAGED_BY, build_chunks, chunk_markdown
from app.rag.models import ChunkingConfig
from app.rag.vector_store import store_chunks


KNOWLEDGE_BASE = Path(__file__).parents[1] / "data" / "knowledge_base"
CONFIG = ChunkingConfig(min_tokens=120, target_tokens=300, max_tokens=450, overlap_tokens=40)


class FakeVectorStore:
    def __init__(self, existing_ids: list[str] | None = None) -> None:
        self.existing_ids = existing_ids or []
        self.added_documents: list[Document] = []
        self.added_ids: list[str] = []
        self.deleted: list[str] = []

    def get(self, **_: object) -> dict[str, list[str]]:
        return {"ids": self.existing_ids}

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.added_documents.extend(documents)
        self.added_ids.extend(ids)

    def delete(self, *, ids: list[str]) -> None:
        self.deleted.extend(ids)


def test_current_corpus_chunks_are_bounded_and_deterministic() -> None:
    chunks = build_chunks(KNOWLEDGE_BASE, CONFIG)
    repeated = build_chunks(KNOWLEDGE_BASE, CONFIG)

    assert len(chunks) >= 20
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in repeated]
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(0 < chunk.metadata["token_count"] <= CONFIG.max_tokens for chunk in chunks)
    assert {chunk.metadata["source_file"] for chunk in chunks} == {
        path.name for path in KNOWLEDGE_BASE.glob("*.md")
    }
    assert all(chunk.metadata["managed_by"] == MANAGED_BY for chunk in chunks)
    assert all(chunk.text.startswith(f"# {chunk.metadata['document_title']}") for chunk in chunks)
    assert all(
        chunk.metadata["primary_section_title"] in chunk.metadata["section_title"]
        for chunk in chunks
    )


def test_nested_headings_and_metadata_are_preserved() -> None:
    markdown = """# Example Guide

## Main Section

### First Method
First details.

### Second Method
Second details.
"""

    chunks = chunk_markdown(
        markdown,
        source_file="example.md",
        config=ChunkingConfig(min_tokens=1, target_tokens=20, max_tokens=30, overlap_tokens=5),
    )

    combined = "\n".join(chunk.text for chunk in chunks)
    assert "### First Method" in combined
    assert "### Second Method" in combined
    assert chunks[0].metadata["document_title"] == "Example Guide"
    assert chunks[0].metadata["section_path"].startswith("Example Guide > Main Section")


def test_forced_token_split_includes_title_within_hard_limit() -> None:
    markdown = "# Long Guide\n\n## Details\n" + ("adaptation recovery volume intensity " * 100)
    config = ChunkingConfig(min_tokens=10, target_tokens=20, max_tokens=30, overlap_tokens=5)

    chunks = chunk_markdown(markdown, source_file="long.md", config=config)

    assert len(chunks) > 1
    assert all(chunk.metadata["token_count"] <= config.max_tokens for chunk in chunks)


def test_store_upserts_chunks_and_removes_only_stale_managed_ids() -> None:
    chunks = build_chunks(KNOWLEDGE_BASE, CONFIG)[:2]
    vector_store = FakeVectorStore(
        existing_ids=[chunks[0].chunk_id, "removed.md::0000"]
    )

    deleted = store_chunks(chunks, vector_store, batch_size=1)

    assert vector_store.added_ids == [chunk.chunk_id for chunk in chunks]
    assert [document.page_content for document in vector_store.added_documents] == [
        chunk.text for chunk in chunks
    ]
    assert deleted == 1
    assert vector_store.deleted == ["removed.md::0000"]
