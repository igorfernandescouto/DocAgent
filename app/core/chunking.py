from __future__ import annotations

from app.core.text import normalize_whitespace, stable_hash
from app.models import Chunk


class TextChunker:
    def __init__(self, chunk_size: int = 140, overlap: int = 30) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document_id: str, source_name: str, text: str) -> list[Chunk]:
        cleaned_text = normalize_whitespace(text)
        words = cleaned_text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            content = " ".join(words[start:end]).strip()
            if not content:
                break

            chunk_id = stable_hash(f"{document_id}:{index}:{content}", length=20)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_name=source_name,
                    content=content,
                    index=index,
                    metadata={"word_count": end - start},
                )
            )

            if end >= len(words):
                break

            start = end - self.overlap
            index += 1

        return chunks
