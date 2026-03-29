from __future__ import annotations

from io import BytesIO
from pathlib import Path

from app.core.text import normalize_whitespace


class DocumentLoader:
    supported_extensions = {".txt", ".md", ".pdf"}

    def load_from_bytes(self, content: bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix in {".txt", ".md"}:
            return normalize_whitespace(content.decode("utf-8", errors="ignore"))

        return self._load_pdf(content)

    def load_from_path(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix in {".txt", ".md"}:
            return normalize_whitespace(path.read_text(encoding="utf-8"))

        return self._load_pdf(path.read_bytes())

    def _load_pdf(self, content: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("pypdf is required to ingest PDF files") from error

        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_whitespace(" ".join(pages))
