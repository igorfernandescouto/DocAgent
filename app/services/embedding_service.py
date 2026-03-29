from __future__ import annotations

from hashlib import sha1
import math
from typing import Protocol

import httpx

from app.core.text import tokenize


class EmbeddingService(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class LocalHashEmbeddingService:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension

        for token in tokenize(text):
            digest = sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] = vector[index] + sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [round(value / norm, 6) for value in vector]


class OpenAICompatibleEmbeddingService:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("data", [])
        if not items:
            raise RuntimeError("Embedding API returned no data")

        return [item["embedding"] for item in items]


class HybridEmbeddingService:
    def __init__(
        self,
        local_service: LocalHashEmbeddingService,
        remote_service: OpenAICompatibleEmbeddingService | None = None,
    ) -> None:
        self.local_service = local_service
        self.remote_service = remote_service

    @property
    def mode(self) -> str:
        return "remote" if self.remote_service else "local"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.remote_service is None:
            return self.local_service.embed_texts(texts)

        try:
            return self.remote_service.embed_texts(texts)
        except Exception:
            return self.local_service.embed_texts(texts)
