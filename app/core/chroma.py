from typing import Any

import chromadb

from app.core.config import Settings


def create_chroma_client(settings: Settings) -> Any:
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        ssl=settings.chroma_ssl,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )
