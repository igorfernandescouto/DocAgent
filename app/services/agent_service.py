from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any
import time

from app.core.evaluation import EvaluationService
from app.core.logging import JsonlExecutionLogger
from app.core.text import normalize_text, similarity_ratio, split_sentences, truncate_text
from app.models import AgentDecision, RetrievedChunk, ToolExecutionResult
from app.services.llm_service import OpenAICompatibleChatService
from app.services.rag_service import RagService
from app.services.tool_service import ToolService


class SimpleResponseCache:
    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        cached = self._items.get(key)
        if cached is None:
            return None

        self._items.move_to_end(key)
        return cached

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


class AgentService:
    def __init__(
        self,
        rag_service: RagService,
        tool_service: ToolService,
        logger: JsonlExecutionLogger,
        evaluation_service: EvaluationService,
        default_top_k: int,
        cache_size: int,
        chat_service: OpenAICompatibleChatService | None = None,
    ) -> None:
        self.rag_service = rag_service
        self.tool_service = tool_service
        self.logger = logger
        self.evaluation_service = evaluation_service
        self.default_top_k = default_top_k
        self.chat_service = chat_service
        self.cache = SimpleResponseCache(max_size=cache_size)

    def handle_question(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        started_at = time.perf_counter()
        retrieval_limit = top_k or self.default_top_k
        rules_decision = self._decide_with_rules(question)
        retrieved_chunks: list[RetrievedChunk] = []
        decision = rules_decision

        if rules_decision.tool_name is None:
            retrieved_chunks = self.rag_service.retrieve(question=question, top_k=retrieval_limit)
            if rules_decision.action == "answer" and self.chat_service is not None:
                llm_decision = self._decide_with_llm(question=question, retrieved_chunks=retrieved_chunks)
                decision = llm_decision or rules_decision

        cache_key = self._cache_key(question=question, top_k=retrieval_limit)
        if decision.tool_name is None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                response = {
                    **cached,
                    "metrics": {
                        **cached["metrics"],
                        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                        "cache_hit": True,
                    },
                }
                self._log_interaction(
                    question=question,
                    action=response["action"],
                    answer=response["answer"],
                    sources=response["sources"],
                    tool_name=None,
                    tool_result=None,
                    metrics=response["metrics"],
                )
                return response

        if decision.tool_name:
            tool_result = self.tool_service.execute(
                tool_name=decision.tool_name,
                question=question,
                payload=decision.tool_input,
            )
            answer = self._render_tool_answer(tool_result)
            sources: list[dict[str, Any]] = []
            used_tool = True
        else:
            answer = self._answer_from_documents(
                question=question,
                retrieved_chunks=retrieved_chunks,
                mode=decision.action,
            )
            sources = self._build_sources(retrieved_chunks)
            tool_result = None
            used_tool = False

        metrics = self.evaluation_service.build_metrics(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            started_at=started_at,
            used_tool=used_tool,
            cache_hit=False,
        )

        response = {
            "answer": answer,
            "action": decision.action,
            "used_tool": used_tool,
            "tool_name": tool_result.tool_name if tool_result else None,
            "tool_result": tool_result.payload if tool_result else None,
            "sources": sources,
            "metrics": metrics,
        }

        self._log_interaction(
            question=question,
            action=decision.action,
            answer=answer,
            sources=sources,
            tool_name=tool_result.tool_name if tool_result else None,
            tool_result=tool_result.payload if tool_result else None,
            metrics=metrics,
        )

        if not used_tool:
            self.cache.set(cache_key, response)

        return response

    def _decide_with_rules(self, question: str) -> AgentDecision:
        normalized = normalize_text(question)

        save_keywords = ["salva", "salvar", "salve", "grave", "registre", "registrar", "cadastre", "cadastrar"]
        external_keywords = ["api", "webhook", "crm", "sistema externo", "integracao externa"]
        summary_keywords = ["resuma", "resumo", "sumarize", "sumarizar"]

        if any(keyword in normalized for keyword in external_keywords):
            return AgentDecision(
                action="tool",
                reason="Pedido explicito de integracao externa",
                tool_name="call_external_api",
            )

        if any(keyword in normalized for keyword in save_keywords):
            return AgentDecision(
                action="tool",
                reason="Pedido explicito para persistir uma informacao",
                tool_name="save_candidate_interest",
            )

        if "interessado" in normalized and "@" in question:
            return AgentDecision(
                action="tool",
                reason="Lead com indicio claro de cadastro",
                tool_name="save_candidate_interest",
            )

        if any(keyword in normalized for keyword in summary_keywords):
            return AgentDecision(action="summarize", reason="Pedido de resumo")

        return AgentDecision(action="answer", reason="Pergunta baseada em documentos")

    def _decide_with_llm(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> AgentDecision | None:
        if self.chat_service is None:
            return None

        preview = "\n".join(item.chunk.content for item in retrieved_chunks[:2]) or "Sem contexto recuperado."
        system_prompt = (
            "Voce decide a proxima acao de um agente simples. "
            "Retorne apenas JSON com as chaves action, reason, tool_name e tool_input. "
            "action deve ser answer, summarize ou tool. "
            "tool_name pode ser save_candidate_interest ou call_external_api."
        )
        user_prompt = (
            f"Pergunta: {question}\n"
            f"Contexto parcial:\n{preview}\n"
            "Exemplo de resposta valida:\n"
            '{"action":"answer","reason":"pergunta documental","tool_name":null,"tool_input":{}}'
        )

        try:
            payload = self.chat_service.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception:
            return None

        action = str(payload.get("action", "answer")).strip().lower()
        if action not in {"answer", "summarize", "tool"}:
            return None

        tool_name = payload.get("tool_name")
        if action != "tool":
            tool_name = None

        if tool_name not in {None, "save_candidate_interest", "call_external_api"}:
            return None

        return AgentDecision(
            action=action,
            reason=str(payload.get("reason", "llm decision")),
            tool_name=tool_name,
            tool_input=payload.get("tool_input") or {},
        )

    def _answer_from_documents(self, question: str, retrieved_chunks: list[RetrievedChunk], mode: str) -> str:
        if not retrieved_chunks:
            return "Nao encontrei informacoes suficientes nos documentos carregados para responder com seguranca."

        if self.chat_service is not None:
            llm_answer = self._answer_with_llm(question=question, retrieved_chunks=retrieved_chunks, mode=mode)
            if llm_answer:
                return llm_answer

        if mode == "summarize":
            return self._extractive_summary(retrieved_chunks)

        return self._extractive_answer(question=question, retrieved_chunks=retrieved_chunks)

    def _answer_with_llm(self, question: str, retrieved_chunks: list[RetrievedChunk], mode: str) -> str | None:
        if self.chat_service is None:
            return None

        context = self.rag_service.build_context(retrieved_chunks)
        task = "gerar um resumo curto e fiel" if mode == "summarize" else "responder a pergunta"
        system_prompt = (
            "Voce responde somente com base no contexto recuperado. "
            "Nao invente fatos. "
            "Se a resposta nao estiver no contexto, diga claramente que nao encontrou a informacao. "
            "Inclua citacoes no formato [arquivo#chunk-n]."
        )
        user_prompt = (
            f"Tarefa: {task}\n"
            f"Pergunta do usuario: {question}\n\n"
            f"Contexto recuperado:\n{context}\n\n"
            "Resposta:"
        )

        try:
            return self.chat_service.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception:
            return None

    def _extractive_answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        ranked_sentences: list[tuple[float, str]] = []
        for item in retrieved_chunks:
            for sentence in split_sentences(item.chunk.content):
                score = similarity_ratio(question, sentence)
                if score == 0:
                    continue
                ranked_sentences.append((score, sentence))

        ranked_sentences.sort(key=lambda entry: entry[0], reverse=True)
        selected = [sentence for _, sentence in ranked_sentences[:3]]
        if not selected:
            selected = [truncate_text(retrieved_chunks[0].chunk.content, limit=260)]

        citations = self._citations(retrieved_chunks)
        return f"{' '.join(selected)} {citations}".strip()

    def _extractive_summary(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        selected: list[str] = []
        seen: set[str] = set()

        for item in retrieved_chunks:
            for sentence in split_sentences(item.chunk.content):
                normalized = normalize_text(sentence)
                if normalized in seen:
                    continue
                selected.append(sentence)
                seen.add(normalized)
                if len(selected) >= 4:
                    citations = self._citations(retrieved_chunks)
                    return f"{' '.join(selected)} {citations}".strip()

        if not selected:
            return "Nao encontrei informacoes suficientes para resumir os documentos."

        citations = self._citations(retrieved_chunks)
        return f"{' '.join(selected)} {citations}".strip()

    def _citations(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in retrieved_chunks[:3]:
            citation = f"[{item.chunk.citation}]"
            if citation in seen:
                continue
            ordered.append(citation)
            seen.add(citation)

        return " ".join(ordered)

    def _render_tool_answer(self, result: ToolExecutionResult) -> str:
        if result.tool_name == "save_candidate_interest":
            name = result.payload.get("name") or "lead sem nome"
            email = result.payload.get("email") or "email nao informado"
            return f"Lead salvo com sucesso: {name} ({email})."

        status = result.payload.get("status") or result.status
        mode = result.payload.get("mode") or "local"
        return f"Integracao executada com status {status} no modo {mode}."

    def _build_sources(self, retrieved_chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for item in retrieved_chunks:
            sources.append(
                {
                    "source_name": item.chunk.source_name,
                    "citation": item.chunk.citation,
                    "chunk_id": item.chunk.chunk_id,
                    "excerpt": truncate_text(item.chunk.content, limit=200),
                    "score": round(item.combined_score, 4),
                }
            )

        return sources

    def _cache_key(self, question: str, top_k: int) -> str:
        normalized = normalize_text(question)
        fingerprint = self.rag_service.document_fingerprint()
        return f"{normalized}|{top_k}|{fingerprint}"

    def _log_interaction(
        self,
        question: str,
        action: str,
        answer: str,
        sources: list[dict[str, Any]],
        tool_name: str | None,
        tool_result: dict[str, Any] | None,
        metrics: dict[str, Any],
    ) -> None:
        self.logger.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "action": action,
                "answer": answer,
                "sources": sources,
                "tool_name": tool_name,
                "tool_result": tool_result,
                "metrics": metrics,
            }
        )
