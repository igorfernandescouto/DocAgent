from __future__ import annotations

import json
from typing import Any

import httpx


class OpenAICompatibleChatService:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("Chat API returned no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join(part for part in text_parts if part).strip()

        raise RuntimeError("Unsupported chat response format")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw = self.complete(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Model did not return JSON")

        return json.loads(raw[start : end + 1])
