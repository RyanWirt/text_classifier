from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def _resolve_path(value: str, default: Path) -> Path:
    candidate = Path(value) if value else default
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int
    data_dir: Path
    data_file_types: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int
    top_k: int
    qdrant_url: str
    qdrant_collection: str
    litellm_base_url: str
    litellm_api_key: str
    litellm_chat_model: str
    litellm_embedding_model: str
    agent_prompt_file: Path


def get_settings() -> Settings:
    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "5000")),
        data_dir=_resolve_path(os.getenv("DATA_DIR", "data"), REPO_ROOT / "data"),
        data_file_types=_split_csv(os.getenv("DATA_FILE_TYPES", ".pdf,.json,.xml,.txt")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "5")),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "dsm_v_reference"),
        litellm_base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
        litellm_api_key=os.getenv("LITELLM_API_KEY", ""),
        litellm_chat_model=os.getenv("LITELLM_CHAT_MODEL", "openai/gpt-4.1-mini"),
        litellm_embedding_model=os.getenv(
            "LITELLM_EMBEDDING_MODEL", "openai/text-embedding-3-small"
        ),
        agent_prompt_file=_resolve_path(
            os.getenv("AGENT_PROMPT_FILE", "prompts/dsm_v_review.txt"),
            REPO_ROOT / "prompts" / "dsm_v_review.txt",
        ),
    )
