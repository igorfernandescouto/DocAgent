from __future__ import annotations

from pathlib import Path

from app.core.chunking import TextChunker
from app.core.text import normalize_whitespace, stable_hash
from app.models import RetrievedChunk
from app.services.document_loader import DocumentLoader
from app.services.embedding_service import HybridEmbeddingService
from app.services.vector_store import JsonVectorStore


class RagService:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedding_service: HybridEmbeddingService,
        vector_store: JsonVectorStore,
        default_top_k: int,
        min_score: float,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.min_score = min_score

    def ingest_text(self, text: str, source_name: str) -> dict[str, int | str]:
        cleaned_text = normalize_whitespace(text)
        if len(cleaned_text) < 20:
            raise ValueError("Document is too short to ingest")

        document_id = stable_hash(f"{source_name}:{cleaned_text}", length=18)
        chunks = self.chunker.chunk_document(document_id=document_id, source_name=source_name, text=cleaned_text)
        embeddings = self.embedding_service.embed_texts([chunk.content for chunk in chunks])
        self.vector_store.upsert(chunks, embeddings)

        return {
            "document_id": document_id,
            "source_name": source_name,
            "chunks_created": len(chunks),
            "characters": len(cleaned_text),
        }

    def ingest_file(self, content: bytes, filename: str, source_name: str | None = None) -> dict[str, int | str]:
        text = self.loader.load_from_bytes(content, filename)
        target_name = source_name or filename
        return self.ingest_text(text=text, source_name=target_name)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self.vector_store.count() == 0:
            return []

        limit = top_k or self.default_top_k
        query_embedding = self.embedding_service.embed_texts([question])[0]
        return self.vector_store.search(
            query_text=question,
            query_embedding=query_embedding,
            top_k=limit,
            min_score=self.min_score,
        )

    def list_documents(self) -> list[dict]:
        return self.vector_store.list_documents()

    def document_fingerprint(self) -> str:
        return self.vector_store.document_fingerprint()

    def build_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for item in retrieved_chunks:
            parts.append(
                "\n".join(
                    [
                        f"Fonte: {item.chunk.source_name}",
                        f"Citation: {item.chunk.citation}",
                        f"Conteudo: {item.chunk.content}",
                    ]
                )
            )

        return "\n\n".join(parts).strip()

    def bootstrap_from_directory(self, directory: Path) -> int:
        if self.vector_store.count() > 0:
            return 0

        ingested = 0
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.loader.supported_extensions:
                continue

            text = self.loader.load_from_path(path)
            self.ingest_text(text=text, source_name=path.name)
            ingested += 1

        return ingested
