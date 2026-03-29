from __future__ import annotations

from statistics import fmean
import time

from app.core.text import estimate_token_count, similarity_ratio
from app.models import RetrievedChunk


class EvaluationService:
    def build_metrics(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        started_at: float,
        used_tool: bool,
        cache_hit: bool,
    ) -> dict[str, float | int | bool]:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        context_text = " ".join(item.chunk.content for item in retrieved_chunks)
        groundedness = similarity_ratio(answer, context_text) if context_text else 0.0
        average_score = fmean([item.combined_score for item in retrieved_chunks]) if retrieved_chunks else 0.0

        return {
            "latency_ms": latency_ms,
            "question_tokens_est": estimate_token_count(question),
            "answer_tokens_est": estimate_token_count(answer),
            "context_tokens_est": estimate_token_count(context_text),
            "total_tokens_est": estimate_token_count(question) + estimate_token_count(answer) + estimate_token_count(context_text),
            "groundedness_score": round(groundedness, 3),
            "average_retrieval_score": round(average_score, 3),
            "used_context": bool(retrieved_chunks),
            "used_tool": used_tool,
            "cache_hit": cache_hit,
        }

    def summarize_logs(self, items: list[dict]) -> dict[str, float | int]:
        if not items:
            return {
                "requests": 0,
                "avg_latency_ms": 0.0,
                "avg_groundedness_score": 0.0,
                "tool_calls": 0,
                "cache_hits": 0,
            }

        metrics = [item.get("metrics", {}) for item in items]
        latencies = [float(metric.get("latency_ms", 0.0)) for metric in metrics]
        groundedness = [float(metric.get("groundedness_score", 0.0)) for metric in metrics]
        tool_calls = sum(1 for metric in metrics if metric.get("used_tool"))
        cache_hits = sum(1 for metric in metrics if metric.get("cache_hit"))

        return {
            "requests": len(items),
            "avg_latency_ms": round(fmean(latencies), 2),
            "avg_groundedness_score": round(fmean(groundedness), 3),
            "tool_calls": tool_calls,
            "cache_hits": cache_hits,
        }
