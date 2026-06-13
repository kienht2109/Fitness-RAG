from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AI Workout Coach"
    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    openai_api_key: SecretStr
    openai_chat_model: str = "gpt-4o-mini"
    openai_agent_model: str = "gpt-4o"
    openai_judge_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"
    chroma_collection: str = "fitness_knowledge"

    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_chunk_min_tokens: int = Field(default=120, ge=1)
    rag_chunk_target_tokens: int = Field(default=300, ge=1)
    rag_chunk_max_tokens: int = Field(default=450, ge=1)
    rag_chunk_overlap_tokens: int = Field(default=40, ge=0)
    rag_embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    agent_max_iterations: int = Field(default=5, ge=1, le=20)
    knowledge_base_dir: Path = Path("data/knowledge_base")


@lru_cache
def get_settings() -> Settings:
    return Settings()
