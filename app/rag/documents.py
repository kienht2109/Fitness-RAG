from pathlib import Path


def load_markdown_files(knowledge_base_dir: Path) -> list[Path]:
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {knowledge_base_dir}")

    files = sorted(path for path in knowledge_base_dir.glob("*.md") if path.is_file())
    if not files:
        raise ValueError(f"No Markdown files found in: {knowledge_base_dir}")
    return files
