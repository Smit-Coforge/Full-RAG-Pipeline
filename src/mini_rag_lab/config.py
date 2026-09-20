from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://mini_rag_lab:mini_rag_lab@db:5432/mini_rag_lab"
    ollama_host: str = "http://ollama:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: Literal[768] = 768
    generation_model: str = "llama3.2:3b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
