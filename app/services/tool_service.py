from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any

import httpx

from app.core.text import stable_hash
from app.models import ToolExecutionResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonListRepository:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def load(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []

        with self.file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return list(payload)

    def save(self, items: list[dict[str, Any]]) -> None:
        with self._lock:
            with self.file_path.open("w", encoding="utf-8") as handle:
                json.dump(items, handle, ensure_ascii=True, indent=2)

    def append(self, item: dict[str, Any]) -> None:
        items = self.load()
        items.append(item)
        self.save(items)


class JsonlFileRepository:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, item: dict[str, Any]) -> None:
        payload = json.dumps(item, ensure_ascii=True)
        with self._lock:
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")


class ToolService:
    def __init__(
        self,
        leads_path: Path,
        external_calls_path: Path,
        external_api_base_url: str,
        timeout_seconds: float,
    ) -> None:
        self.external_api_base_url = external_api_base_url
        self.timeout_seconds = timeout_seconds
        self.leads_repository = JsonListRepository(leads_path)
        self.external_calls_repository = JsonlFileRepository(external_calls_path)

    def available_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": "save_candidate_interest",
                "description": "Salva um cliente ou lead interessado em arquivo JSON.",
            },
            {
                "name": "call_external_api",
                "description": "Envia payload para uma API externa ou simula a chamada localmente.",
            },
        ]

    def execute(self, tool_name: str, question: str, payload: dict[str, Any] | None = None) -> ToolExecutionResult:
        if tool_name == "save_candidate_interest":
            return self._save_candidate_interest(question=question, payload=payload or {})

        if tool_name == "call_external_api":
            return self._call_external_api(question=question, payload=payload or {})

        raise ValueError(f"Unknown tool: {tool_name}")

    def _save_candidate_interest(self, question: str, payload: dict[str, Any]) -> ToolExecutionResult:
        prepared = self._prepare_lead_payload(question=question, payload=payload)
        lead_id = stable_hash(f"{prepared.get('name', '')}:{prepared.get('email', '')}:{question}", length=12)
        entry = {
            "lead_id": lead_id,
            "saved_at": _utc_now(),
            **prepared,
        }
        self.leads_repository.append(entry)

        return ToolExecutionResult(
            tool_name="save_candidate_interest",
            status="saved",
            payload=entry,
        )

    def _call_external_api(self, question: str, payload: dict[str, Any]) -> ToolExecutionResult:
        prepared = self._prepare_api_payload(question=question, payload=payload)
        record = {
            "request_id": stable_hash(f"{question}:{_utc_now()}", length=12),
            "executed_at": _utc_now(),
            "payload": prepared,
        }

        if not self.external_api_base_url:
            record["mode"] = "simulated"
            record["status"] = "queued_locally"
            self.external_calls_repository.append(record)
            return ToolExecutionResult(
                tool_name="call_external_api",
                status="simulated",
                payload=record,
            )

        try:
            response = httpx.post(
                f"{self.external_api_base_url}/leads",
                json=prepared,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            try:
                response_payload: Any = response.json()
            except ValueError:
                response_payload = {"raw_response": response.text}

            record["mode"] = "remote"
            record["status"] = "sent"
            record["http_status"] = response.status_code
            record["response"] = response_payload
            self.external_calls_repository.append(record)
            return ToolExecutionResult(
                tool_name="call_external_api",
                status="sent",
                payload=record,
            )
        except Exception as error:
            record["mode"] = "fallback"
            record["status"] = "queued_locally"
            record["error"] = str(error)
            self.external_calls_repository.append(record)
            return ToolExecutionResult(
                tool_name="call_external_api",
                status="queued_locally",
                payload=record,
            )

    def _prepare_lead_payload(self, question: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": payload.get("name")
            or self._extract_pattern(question, r"nome[:=]\s*([a-zA-Z ]+?)(?=\s+(?:email|telefone|empresa)[:=]|\s*$)"),
            "email": payload.get("email") or self._extract_pattern(question, r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"),
            "phone": payload.get("phone") or self._extract_pattern(question, r"(\+?\d[\d\s-]{7,}\d)"),
            "company": payload.get("company")
            or self._extract_pattern(question, r"empresa[:=]\s*([a-zA-Z0-9 ]+?)(?=\s+(?:email|telefone|nome)[:=]|\s*$)"),
            "notes": payload.get("notes") or question.strip(),
        }

    def _prepare_api_payload(self, question: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "message": payload.get("message") or question.strip(),
            "source": payload.get("source") or "docagent",
            "contact_email": payload.get("contact_email") or self._extract_pattern(
                question, r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
            ),
            "metadata": payload.get("metadata") or {},
        }

    @staticmethod
    def _extract_pattern(text: str, pattern: str) -> str | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None

        return match.group(1).strip()
