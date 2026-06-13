from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    min_tokens: int = 120
    target_tokens: int = 300
    max_tokens: int = 450
    overlap_tokens: int = 40

    def __post_init__(self) -> None:
        if not 0 < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("Chunk token limits must satisfy 0 < min <= target <= max")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("Chunk overlap must satisfy 0 <= overlap < max")


@dataclass(frozen=True)
class MarkdownUnit:
    section_title: str
    section_path: str
    markdown: str


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class IngestionReport:
    source_files: int
    chunks_upserted: int
    stale_chunks_deleted: int


@dataclass(frozen=True)
class RetrievalSource:
    source_file: str
    section_title: str
    chunk_id: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    answer: str
    sources: list[RetrievalSource]
