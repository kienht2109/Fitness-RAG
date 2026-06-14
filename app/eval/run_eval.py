"""Run the live evaluation suite and write machine-readable results."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable

from app.agent.orchestrator import AgentService, create_agent_service
from app.analysis.insight import AnalysisService, create_analysis_service
from app.core.config import Settings, get_settings
from app.eval.metrics import (
    FaithfulnessJudge,
    MetricResult,
    create_faithfulness_judge,
    data_grounding,
    expected_data_points,
    expected_source_present,
    format_evidence,
    guardrail_correctness,
    source_attribution,
    tool_selection,
)
from app.eval.test_set import (
    AgentCase,
    AnalysisCase,
    EvaluationTestSet,
    GuardrailCase,
    RagCase,
    load_test_set,
)
from app.rag.chunking import build_chunks
from app.rag.ingest import chunking_config
from app.rag.retrieve import RetrievalService, create_retrieval_service

DEFAULT_OUTPUT_PATH = Path("evaluation-results.json")
DEFAULT_TEST_SET_PATH = Path("evaluation-test-set.json")
FAITHFULNESS_PASS_SCORE = 4


class EvaluationRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        retrieval_service: RetrievalService,
        analysis_service: AnalysisService,
        agent_service: AgentService,
        judge: FaithfulnessJudge,
        test_set: EvaluationTestSet,
        test_set_path: Path,
    ) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.agent_service = agent_service
        self.judge = judge
        self.test_set = test_set
        self.test_set_path = test_set_path
        self.known_chunk_ids = {
            chunk.chunk_id
            for chunk in build_chunks(settings.knowledge_base_dir, chunking_config(settings))
        }

    async def run(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        results: list[dict[str, Any]] = []
        for case in self.test_set.rag_cases:
            results.append(await self._capture(case.case_id, "rag", lambda: self._run_rag(case)))
        for case in self.test_set.analysis_cases:
            results.append(
                await self._capture(case.case_id, "analysis", lambda: self._run_analysis(case))
            )
        for case in self.test_set.agent_cases:
            results.append(
                await self._capture(case.case_id, "agent", lambda: self._run_agent(case))
            )
        for case in self.test_set.guardrail_cases:
            results.append(
                await self._capture(
                    case.case_id, "guardrail", lambda: self._run_guardrail(case)
                )
            )

        finished_at = datetime.now(UTC)
        return {
            "generated_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "configuration": {
                "chat_model": self.settings.openai_chat_model,
                "agent_model": self.settings.openai_agent_model,
                "judge_model": self.settings.openai_judge_model,
                "embedding_model": self.settings.openai_embedding_model,
                "faithfulness_pass_score": FAITHFULNESS_PASS_SCORE,
                "test_set_path": str(self.test_set_path),
            },
            "aggregate": _aggregate(results),
            "cases": results,
        }

    async def _run_rag(self, case: RagCase) -> dict[str, Any]:
        result = await self.retrieval_service.query(case.question)
        metrics = {
            "source_attribution": source_attribution(
                result.sources,
                knowledge_base_dir=self.settings.knowledge_base_dir,
                known_chunk_ids=self.known_chunk_ids,
            ),
            "expected_source": expected_source_present(
                result.sources, case.expected_source_doc
            ),
            "guardrail_correctness": guardrail_correctness(
                result.answer, should_refuse=False
            ),
        }
        judgment = await self.judge.evaluate(
            question=case.question,
            answer=result.answer,
            evidence=result.context,
        )
        return {
            "input": case.model_dump(mode="json"),
            "output": {
                "answer": result.answer,
                "sources": [source.__dict__ for source in result.sources],
            },
            "metrics": {
                **_metric_dicts(metrics),
                "faithfulness": {
                    "passed": judgment.score >= FAITHFULNESS_PASS_SCORE,
                    "score": judgment.score,
                    "rationale": judgment.rationale,
                },
            },
        }

    async def _run_analysis(self, case: AnalysisCase) -> dict[str, Any]:
        result = await self.analysis_service.query(user_id=case.user_id, question=case.question)
        expected = [(point.path, point.value) for point in case.expected_data_points]
        metrics = {
            "data_grounding": data_grounding(result.insight, result.summary),
            "expected_data_points": expected_data_points(result.summary, expected),
        }
        return {
            "input": case.model_dump(mode="json"),
            "output": {"insight": result.insight, "summary": result.summary},
            "metrics": _metric_dicts(metrics),
        }

    async def _run_agent(self, case: AgentCase) -> dict[str, Any]:
        result = await self.agent_service.query(user_id=case.user_id, question=case.question)
        metrics = {"tool_selection": tool_selection(result.tools_used, case.expected_tools)}
        judgment = await self.judge.evaluate(
            question=case.question,
            answer=result.answer,
            evidence=format_evidence(result.tool_outputs),
        )
        return {
            "input": case.model_dump(mode="json"),
            "output": {
                "answer": result.answer,
                "tools_used": result.tools_used,
                "tool_outputs": result.tool_outputs,
            },
            "metrics": {
                **_metric_dicts(metrics),
                "faithfulness": {
                    "passed": judgment.score >= FAITHFULNESS_PASS_SCORE,
                    "score": judgment.score,
                    "rationale": judgment.rationale,
                },
            },
        }

    async def _run_guardrail(self, case: GuardrailCase) -> dict[str, Any]:
        result = await self.retrieval_service.query(case.question)
        metric = guardrail_correctness(
            result.answer,
            should_refuse=True,
            expected_category=case.expected_category,
        )
        return {
            "input": case.model_dump(mode="json"),
            "output": {"answer": result.answer, "sources": []},
            "metrics": {"guardrail_correctness": metric.to_dict()},
        }

    async def _capture(
        self,
        case_id: str,
        category: str,
        operation: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        started = perf_counter()
        try:
            payload = await operation()
            metrics = payload.get("metrics", {})
            passed = bool(metrics) and all(metric.get("passed") is True for metric in metrics.values())
            return {
                "case_id": case_id,
                "category": category,
                "passed": passed,
                "duration_seconds": round(perf_counter() - started, 3),
                **payload,
            }
        except Exception as exc:
            return {
                "case_id": case_id,
                "category": category,
                "passed": False,
                "duration_seconds": round(perf_counter() - started, 3),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }


def create_runner(
    settings: Settings | None = None,
    *,
    test_set_path: Path = DEFAULT_TEST_SET_PATH,
) -> EvaluationRunner:
    test_set = load_test_set(test_set_path)
    settings = settings or get_settings()
    return EvaluationRunner(
        settings=settings,
        retrieval_service=create_retrieval_service(settings),
        analysis_service=create_analysis_service(settings),
        agent_service=create_agent_service(settings),
        judge=create_faithfulness_judge(settings),
        test_set=test_set,
        test_set_path=test_set_path,
    )


def _metric_dicts(metrics: dict[str, MetricResult]) -> dict[str, dict[str, Any]]:
    return {name: result.to_dict() for name, result in metrics.items()}


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_totals: dict[str, dict[str, float]] = {}
    category_totals: dict[str, dict[str, int]] = {}
    faithfulness_scores: list[int] = []
    for result in results:
        category = result["category"]
        category_stats = category_totals.setdefault(category, {"total": 0, "passed": 0})
        category_stats["total"] += 1
        category_stats["passed"] += int(result["passed"])
        for name, metric in result.get("metrics", {}).items():
            stats = metric_totals.setdefault(name, {"total": 0, "passed": 0})
            stats["total"] += 1
            stats["passed"] += int(metric.get("passed") is True)
            if name == "faithfulness" and isinstance(metric.get("score"), int):
                faithfulness_scores.append(metric["score"])

    for stats in metric_totals.values():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 4)
    for stats in category_totals.values():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 4)

    return {
        "total_cases": len(results),
        "passed_cases": sum(int(result["passed"]) for result in results),
        "failed_cases": sum(int(not result["passed"]) for result in results),
        "case_pass_rate": round(
            sum(int(result["passed"]) for result in results) / len(results), 4
        ),
        "average_faithfulness_score": (
            round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
            if faithfulness_scores
            else None
        ),
        "by_category": category_totals,
        "by_metric": metric_totals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Workout Coach evaluation suite")
    parser.add_argument(
        "--test-set",
        type=Path,
        default=DEFAULT_TEST_SET_PATH,
        help=f"JSON test-set path (default: {DEFAULT_TEST_SET_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON results path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


async def async_main(output_path: Path, test_set_path: Path) -> dict[str, Any]:
    results = await create_runner(test_set_path=test_set_path).run()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    args = parse_args()
    results = asyncio.run(async_main(args.output, args.test_set))
    aggregate = results["aggregate"]
    print(
        f"Evaluation complete: {aggregate['passed_cases']}/{aggregate['total_cases']} cases passed. "
        f"Results: {args.output}"
    )


if __name__ == "__main__":
    main()
