from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from app.core.text import filtered_tokens, keyword_overlap
from app.models import Chunk, RetrievedChunk


class JsonVectorStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            self._records = []
            return

        with self.file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self._records = list(payload.get("items", []))

    def _save(self) -> None:
        payload = {"version": 1, "items": self._records}
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def count(self) -> int:
        return len(self._records)

    def list_documents(self) -> list[dict]:
        documents: dict[str, dict] = {}
        for record in self._records:
            chunk = record["chunk"]
            document_id = chunk["document_id"]
            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "source_name": chunk["source_name"],
                    "chunks": 0,
                }

            documents[document_id]["chunks"] += 1

        return list(documents.values())

    def document_fingerprint(self) -> str:
        items = [record["chunk"]["document_id"] for record in self._records]
        return "|".join(sorted(items))

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        if not chunks:
            return

        document_id = chunks[0].document_id
        records = [
            {
                "chunk": chunk.to_dict(),
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        with self._lock:
            self._records = [record for record in self._records if record["chunk"]["document_id"] != document_id]
            self._records.extend(records)
            self._save()

    def search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 4,
        min_score: float = 0.18,
    ) -> list[RetrievedChunk]:
        if not self._records:
            return []

        query_tokens = filtered_tokens(query_text)
        ranked: list[RetrievedChunk] = []

        for record in self._records:
            chunk = Chunk.from_dict(record["chunk"])
            embedding = [float(value) for value in record["embedding"]]
            vector_score = self._cosine_similarity(query_embedding, embedding)
            keyword_score = keyword_overlap(query_tokens, filtered_tokens(chunk.content))
            combined_score = round((0.75 * vector_score) + (0.25 * keyword_score), 4)

            ranked.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(vector_score, 4),
                    keyword_score=round(keyword_score, 4),
                    combined_score=combined_score,
                )
            )

        ranked.sort(key=lambda item: item.combined_score, reverse=True)
        selected = [item for item in ranked if item.combined_score >= min_score]
        if selected:
            return selected[:top_k]

        return ranked[:top_k]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0

        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0

        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        return float(numerator / (left_norm * right_norm))
