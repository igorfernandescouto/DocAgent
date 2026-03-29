from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.container import AppContainer


SAMPLE_CASES = [
    {
        "name": "policy_question",
        "question": "Qual e a politica de devolucao para um produto sem defeito?",
        "expected_keywords": ["7 dias", "frete", "cliente"],
        "expected_action": "answer",
    },
    {
        "name": "summary_request",
        "question": "Resuma o Plano Premium",
        "expected_keywords": ["Plano Premium", "suporte prioritario"],
        "expected_action": "summarize",
    },
    {
        "name": "tool_save",
        "question": "Salve esse cliente como interessado. Nome: Ana Silva Email: ana@empresa.com Telefone: 11999999999",
        "expected_keywords": ["Lead salvo"],
        "expected_action": "tool",
    },
]


def main() -> None:
    container = AppContainer()
    container.bootstrap()

    results: list[dict] = []
    for case in SAMPLE_CASES:
        response = container.agent_service.handle_question(case["question"])
        answer = response["answer"]
        passed_keywords = all(keyword.lower() in answer.lower() for keyword in case["expected_keywords"])
        passed_action = response["action"] == case["expected_action"]

        results.append(
            {
                "case": case["name"],
                "question": case["question"],
                "answer": answer,
                "action": response["action"],
                "passed_keywords": passed_keywords,
                "passed_action": passed_action,
                "metrics": response["metrics"],
            }
        )

    summary = {
        "cases": len(results),
        "passed": sum(1 for result in results if result["passed_keywords"] and result["passed_action"]),
        "avg_latency_ms": round(
            sum(float(result["metrics"]["latency_ms"]) for result in results) / len(results),
            2,
        ),
        "avg_groundedness_score": round(
            sum(float(result["metrics"]["groundedness_score"]) for result in results) / len(results),
            3,
        ),
    }

    print(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
