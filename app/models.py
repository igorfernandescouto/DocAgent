from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, description="User question or command")
    session_id: str | None = Field(default=None, description="Optional session id")
    top_k: int | None = Field(default=None, ge=1, le=10, description="How many chunks to retrieve")


class SourceReference(BaseModel):
    source_name: str
    citation: str
    chunk_id: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    answer: str
    action: str
    used_tool: bool
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None
    sources: list[SourceReference]
    metrics: dict[str, Any]


class IngestResponse(BaseModel):
    document_id: str
    source_name: str
    chunks_created: int
    characters: int


class LogsResponse(BaseModel):
    total: int
    items: list[dict[str, Any]]
    summary: dict[str, Any]


class DocumentsResponse(BaseModel):
    total: int
    items: list[dict[str, Any]]


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source_name: str
    content: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source_name}#chunk-{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            source_name=data["source_name"],
            content=data["content"],
            index=int(data["index"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    keyword_score: float
    combined_score: float


@dataclass
class AgentDecision:
    action: str
    reason: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    tool_name: str
    status: str
    payload: dict[str, Any]
