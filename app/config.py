from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    return int(value)


def _as_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    return float(value)


def _resolve_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str
    host: str
    port: int
    top_k: int
    min_score: float
    cache_size: int
    embedding_dimension: int
    request_timeout_seconds: float
    openai_api_key: str
    openai_base_url: str
    openai_chat_model: str
    openai_embedding_model: str
    external_api_base_url: str
    data_dir: Path
    index_path: Path
    logs_path: Path
    leads_path: Path
    external_calls_path: Path
    sample_docs_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _resolve_path(os.getenv("DOCAGENT_DATA_DIR", "data"))

        return cls(
            app_name=os.getenv("DOCAGENT_APP_NAME", "DocAgent"),
            host=os.getenv("DOCAGENT_HOST", "0.0.0.0"),
            port=_as_int("DOCAGENT_PORT", 8000),
            top_k=_as_int("DOCAGENT_TOP_K", 4),
            min_score=_as_float("DOCAGENT_MIN_SCORE", 0.18),
            cache_size=_as_int("DOCAGENT_CACHE_SIZE", 100),
            embedding_dimension=_as_int("DOCAGENT_EMBEDDING_DIMENSION", 384),
            request_timeout_seconds=_as_float("DOCAGENT_TIMEOUT_SECONDS", 20.0),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "").strip(),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "").strip(),
            external_api_base_url=os.getenv("EXTERNAL_API_BASE_URL", "").rstrip("/"),
            data_dir=data_dir,
            index_path=(data_dir / "index" / "vector_store.json").resolve(),
            logs_path=(data_dir / "logs" / "executions.jsonl").resolve(),
            leads_path=(data_dir / "tool_storage" / "interested_leads.json").resolve(),
            external_calls_path=(data_dir / "tool_storage" / "external_api_calls.jsonl").resolve(),
            sample_docs_path=(data_dir / "sample_docs").resolve(),
        )

    @property
    def chat_enabled(self) -> bool:
        return bool(self.openai_api_key and self.openai_chat_model)

    @property
    def remote_embeddings_enabled(self) -> bool:
        return bool(self.openai_api_key and self.openai_embedding_model)

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.index_path.parent,
            self.logs_path.parent,
            self.leads_path.parent,
            self.sample_docs_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)
