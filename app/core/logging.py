from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlExecutionLogger:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, entry: dict[str, Any]) -> None:
        payload = json.dumps(entry, ensure_ascii=True)
        with self._lock:
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")

    def read(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []

        with self.file_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]

        parsed = [json.loads(line) for line in lines]
        return parsed[-limit:]
