from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from app.container import AppContainer
from app.models import AskRequest, AskResponse, DocumentsResponse, IngestResponse, LogsResponse


def create_app() -> FastAPI:
    container = AppContainer()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bootstrapped_documents = container.bootstrap()
        app.state.container = container
        app.state.bootstrapped_documents = bootstrapped_documents
        yield

    app = FastAPI(
        title="DocAgent",
        version="1.0.0",
        description="Simple RAG + tools API for document-grounded answers and automation.",
        lifespan=lifespan,
    )

    def get_container(request: Request) -> AppContainer:
        return request.app.state.container

    @app.get("/health")
    async def health(request: Request) -> dict:
        current = get_container(request)
        return {
            "status": "ok",
            "documents": len(current.rag_service.list_documents()),
            "chunks": current.vector_store.count(),
            "chat_enabled": current.settings.chat_enabled,
            "embedding_mode": current.embedding_service.mode,
            "external_api_configured": bool(current.settings.external_api_base_url),
            "bootstrapped_documents": getattr(request.app.state, "bootstrapped_documents", 0),
        }

    @app.get("/documents", response_model=DocumentsResponse)
    async def documents(request: Request) -> DocumentsResponse:
        current = get_container(request)
        items = current.rag_service.list_documents()
        return DocumentsResponse(total=len(items), items=items)

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        request: Request,
        file: UploadFile | None = File(default=None),
        text: str | None = Form(default=None),
        source_name: str | None = Form(default=None),
    ) -> IngestResponse:
        current = get_container(request)

        if file is None and not text:
            raise HTTPException(status_code=400, detail="Send a file or a text field.")

        if file is not None:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            result = current.rag_service.ingest_file(
                content=content,
                filename=file.filename or "uploaded_document.txt",
                source_name=source_name,
            )
            return IngestResponse(**result)

        result = current.rag_service.ingest_text(
            text=text or "",
            source_name=source_name or "manual_input.txt",
        )
        return IngestResponse(**result)

    @app.post("/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        current = get_container(request)
        response = current.agent_service.handle_question(
            question=payload.question,
            top_k=payload.top_k,
        )
        return AskResponse(**response)

    @app.get("/logs", response_model=LogsResponse)
    async def logs(request: Request, limit: int = 20) -> LogsResponse:
        current = get_container(request)
        items = current.logger.read(limit=limit)
        summary = current.evaluation_service.summarize_logs(items)
        return LogsResponse(total=len(items), items=items, summary=summary)

    @app.get("/metrics")
    async def metrics(request: Request, limit: int = 200) -> dict:
        current = get_container(request)
        items = current.logger.read(limit=limit)
        return current.evaluation_service.summarize_logs(items)

    return app
