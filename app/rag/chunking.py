import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

from app.rag.documents import load_markdown_files
from app.rag.models import ChunkingConfig, KnowledgeChunk, MarkdownUnit
from app.rag.tokenization import TokenCounter


MANAGED_BY = "fitness_rag_ingest"
CHUNK_PREFIX_TOKEN_RESERVE = 4
HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def _document_title(markdown: str, source_file: str) -> str:
    for line in markdown.splitlines():
        match = HEADING_PATTERN.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return Path(source_file).stem.replace("-", " ").title()


def _h2_sections(markdown: str, document_title: str) -> list[MarkdownUnit]:
    lines = markdown.splitlines()
    sections: list[MarkdownUnit] = []
    current_title = "Overview"
    current_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                MarkdownUnit(
                    section_title=current_title,
                    section_path=f"{document_title} > {current_title}",
                    markdown=content,
                )
            )

    for line in lines:
        match = HEADING_PATTERN.match(line)
        level = len(match.group(1)) if match else None
        if level == 1:
            continue
        if level == 2 and match:
            flush()
            current_title = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    return sections


def _split_at_h3(unit: MarkdownUnit) -> list[MarkdownUnit]:
    lines = unit.markdown.splitlines()
    parent_heading = lines[0] if lines and lines[0].startswith("## ") else f"## {unit.section_title}"
    intro_lines: list[str] = []
    children: list[MarkdownUnit] = []
    child_title: str | None = None
    child_lines: list[str] = []

    def flush_child() -> None:
        if child_title is None:
            return
        children.append(
            MarkdownUnit(
                section_title=child_title,
                section_path=f"{unit.section_path} > {child_title}",
                markdown="\n".join([parent_heading, *child_lines]).strip(),
            )
        )

    for line in lines[1:]:
        match = HEADING_PATTERN.match(line)
        if match and len(match.group(1)) == 3:
            flush_child()
            child_title = match.group(2).strip()
            child_lines = [line]
        elif child_title is None:
            intro_lines.append(line)
        else:
            child_lines.append(line)
    flush_child()

    intro = "\n".join([parent_heading, *intro_lines]).strip()
    if intro_lines and intro:
        children.insert(
            0,
            MarkdownUnit(
                section_title=unit.section_title,
                section_path=unit.section_path,
                markdown=intro,
            ),
        )
    return children or [unit]


def _split_oversized_unit(
    unit: MarkdownUnit,
    counter: TokenCounter,
    config: ChunkingConfig,
) -> list[MarkdownUnit]:
    if counter.count(unit.markdown) <= config.max_tokens:
        return [unit]

    h3_units = _split_at_h3(unit)
    if len(h3_units) > 1:
        return [
            split_unit
            for h3_unit in h3_units
            for split_unit in _split_oversized_unit(h3_unit, counter, config)
        ]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", unit.markdown) if part.strip()]
    if len(paragraphs) > 1:
        pieces: list[MarkdownUnit] = []
        for paragraph in paragraphs:
            texts = (
                [paragraph]
                if counter.count(paragraph) <= config.max_tokens
                else counter.split(paragraph, config.max_tokens, config.overlap_tokens)
            )
            pieces.extend(
                MarkdownUnit(unit.section_title, unit.section_path, text) for text in texts
            )
        return pieces

    return [
        MarkdownUnit(unit.section_title, unit.section_path, text)
        for text in counter.split(unit.markdown, config.max_tokens, config.overlap_tokens)
    ]


def _render_chunk(document_title: str, units: Sequence[MarkdownUnit]) -> str:
    return "\n\n".join([f"# {document_title}", *(unit.markdown for unit in units)]).strip()


def _partition_units(
    document_title: str,
    units: Sequence[MarkdownUnit],
    counter: TokenCounter,
    config: ChunkingConfig,
) -> list[list[MarkdownUnit]]:
    groups: list[list[MarkdownUnit]] = []
    current: list[MarkdownUnit] = []

    for unit in units:
        candidate = [*current, unit]
        candidate_tokens = counter.count(_render_chunk(document_title, candidate))
        current_tokens = counter.count(_render_chunk(document_title, current)) if current else 0
        if current and current_tokens >= config.min_tokens and candidate_tokens > config.target_tokens:
            groups.append(current)
            current = [unit]
        elif current and candidate_tokens > config.max_tokens:
            groups.append(current)
            current = [unit]
        else:
            current = candidate

    if current:
        groups.append(current)

    if len(groups) > 1:
        last_tokens = counter.count(_render_chunk(document_title, groups[-1]))
        while last_tokens < config.min_tokens and len(groups[-2]) > 1:
            groups[-1].insert(0, groups[-2].pop())
            last_tokens = counter.count(_render_chunk(document_title, groups[-1]))
        if last_tokens < config.min_tokens:
            merged = [*groups[-2], *groups[-1]]
            if counter.count(_render_chunk(document_title, merged)) <= config.max_tokens:
                groups[-2:] = [merged]

    return groups


def chunk_markdown(
    markdown: str,
    source_file: str,
    config: ChunkingConfig,
    counter: TokenCounter | None = None,
) -> list[KnowledgeChunk]:
    counter = counter or TokenCounter()
    document_title = _document_title(markdown, source_file)
    base_units = _h2_sections(markdown, document_title)
    if not base_units:
        raise ValueError(f"Markdown source contains no chunkable content: {source_file}")

    content_max_tokens = (
        config.max_tokens
        - counter.count(f"# {document_title}\n\n")
        - CHUNK_PREFIX_TOKEN_RESERVE
    )
    if content_max_tokens < 1:
        raise ValueError(f"Document title leaves no room under the chunk limit for: {source_file}")

    content_config = ChunkingConfig(
        min_tokens=max(1, min(config.min_tokens, content_max_tokens)),
        target_tokens=max(1, min(config.target_tokens, content_max_tokens)),
        max_tokens=content_max_tokens,
        overlap_tokens=min(config.overlap_tokens, content_max_tokens - 1),
    )
    units = [
        split_unit
        for unit in base_units
        for split_unit in _split_oversized_unit(unit, counter, content_config)
    ]
    groups = _partition_units(document_title, units, counter, config)
    source_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    chunks: list[KnowledgeChunk] = []
    for index, group in enumerate(groups):
        text = _render_chunk(document_title, group)
        section_titles = [unit.section_title for unit in group]
        section_paths = [unit.section_path for unit in group]
        display_section_title = " | ".join(section_titles)
        chunk_id = f"{source_file}::{index:04d}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    "source_file": source_file,
                    "document_title": document_title,
                    "section_title": display_section_title,
                    "primary_section_title": section_titles[0],
                    "section_titles": json.dumps(section_titles, ensure_ascii=True),
                    "section_path": f"{document_title} > {display_section_title}",
                    "section_paths": json.dumps(section_paths, ensure_ascii=True),
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "token_count": counter.count(text),
                    "source_sha256": source_sha256,
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "managed_by": MANAGED_BY,
                },
            )
        )
    return chunks


def build_chunks(
    knowledge_base_dir: Path,
    config: ChunkingConfig,
    counter: TokenCounter | None = None,
) -> list[KnowledgeChunk]:
    counter = counter or TokenCounter()
    return [
        chunk
        for path in load_markdown_files(knowledge_base_dir)
        for chunk in chunk_markdown(
            markdown=path.read_text(encoding="utf-8"),
            source_file=path.name,
            config=config,
            counter=counter,
        )
    ]
