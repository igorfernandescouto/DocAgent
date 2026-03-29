from __future__ import annotations

from app.config import Settings
from app.core.chunking import TextChunker
from app.core.evaluation import EvaluationService
from app.core.logging import JsonlExecutionLogger
from app.services.agent_service import AgentService
from app.services.document_loader import DocumentLoader
from app.services.embedding_service import (
    HybridEmbeddingService,
    LocalHashEmbeddingService,
    OpenAICompatibleEmbeddingService,
)
from app.services.llm_service import OpenAICompatibleChatService
from app.services.rag_service import RagService
from app.services.tool_service import ToolService
from app.services.vector_store import JsonVectorStore


class AppContainer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_directories()

        local_embeddings = LocalHashEmbeddingService(dimension=self.settings.embedding_dimension)
        remote_embeddings = None
        if self.settings.remote_embeddings_enabled:
            remote_embeddings = OpenAICompatibleEmbeddingService(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                model=self.settings.openai_embedding_model,
                timeout_seconds=self.settings.request_timeout_seconds,
            )

        self.embedding_service = HybridEmbeddingService(
            local_service=local_embeddings,
            remote_service=remote_embeddings,
        )
        self.chat_service = None
        if self.settings.chat_enabled:
            self.chat_service = OpenAICompatibleChatService(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                model=self.settings.openai_chat_model,
                timeout_seconds=self.settings.request_timeout_seconds,
            )

        self.logger = JsonlExecutionLogger(self.settings.logs_path)
        self.evaluation_service = EvaluationService()
        self.document_loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=140, overlap=30)
        self.vector_store = JsonVectorStore(self.settings.index_path)
        self.rag_service = RagService(
            loader=self.document_loader,
            chunker=self.chunker,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            default_top_k=self.settings.top_k,
            min_score=self.settings.min_score,
        )
        self.tool_service = ToolService(
            leads_path=self.settings.leads_path,
            external_calls_path=self.settings.external_calls_path,
            external_api_base_url=self.settings.external_api_base_url,
            timeout_seconds=self.settings.request_timeout_seconds,
        )
        self.agent_service = AgentService(
            rag_service=self.rag_service,
            tool_service=self.tool_service,
            logger=self.logger,
            evaluation_service=self.evaluation_service,
            default_top_k=self.settings.top_k,
            cache_size=self.settings.cache_size,
            chat_service=self.chat_service,
        )

    def bootstrap(self) -> int:
        return self.rag_service.bootstrap_from_directory(self.settings.sample_docs_path)
