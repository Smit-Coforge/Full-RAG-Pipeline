from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://mini_rag_lab:mini_rag_lab@db:5432/mini_rag_lab"
    ollama_host: str = "http://host.docker.internal:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: Literal[768] = 768
    generation_model: str = "qwen3:8b"
    generation_thinking: bool = False
    max_cosine_distance: float = Field(default=0.4, ge=0, le=2)


@lru_cache
def get_settings() -> Settings:
    return Settings()
