from __future__ import annotations

from hashlib import sha1
import math
import re


STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "esse",
    "esta",
    "este",
    "isso",
    "mais",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "qual",
    "que",
    "se",
    "sem",
    "ser",
    "sua",
    "suas",
    "suo",
    "sua",
    "um",
    "uma",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    return normalize_whitespace(text).lower()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-z0-9]{2,}", normalized)


def stable_hash(value: str, length: int = 16) -> str:
    digest = sha1(value.encode("utf-8")).hexdigest()
    return digest[:length]


def truncate_text(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text

    return text[: limit - 3].rstrip() + "..."


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def keyword_overlap(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0

    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    if not union:
        return 0.0

    return len(intersection) / len(union)


def filtered_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def similarity_ratio(text_a: str, text_b: str) -> float:
    left = filtered_tokens(text_a)
    right = filtered_tokens(text_b)
    return keyword_overlap(left, right)


def estimate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))
